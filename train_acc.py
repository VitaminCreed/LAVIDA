import argparse
import os
import shutil
import sys
import time
from functools import partial

import deepspeed
import numpy as np
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoProcessor,  get_linear_schedule_with_warmup, set_seed, Qwen2VLConfig, CLIPTextConfig
import torchvision.transforms.v2 as transforms
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from sklearn.metrics import average_precision_score, roc_auc_score

from model import LavidaForCausalLM, LavidaConfig
from train.utils import (AverageMeter, ProgressMeter, dict_to_cuda, VideoResultSummary)
from dataset import UBnormalDataset, ShanghaiTechDataset, AvenueDataset, HybridDataset, collate_fn
from train import YamlArgs
from accelerate import Accelerator, DeepSpeedPlugin


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"  # Need to be adjusted according to the GPU

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)

def load_args(args_file='./train/train_para.yaml'):
    yaml_to_args = YamlArgs(args_file)
    parser = argparse.ArgumentParser(description="Training script")
    parser = yaml_to_args.add_to_parser(parser)
    args = parser.parse_args()
    return args


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def main(args):
    setup_seed(args.random_seed)
    
    deepspeed_plugin = DeepSpeedPlugin(
        gradient_accumulation_steps=args.grad_accumulation_steps,
        gradient_clipping=1.0,
        zero_stage=2,
        zero3_save_16bit_model=False, 
    )

    accelerator = Accelerator(
        deepspeed_plugin=deepspeed_plugin,
        mixed_precision=args.precision, 
    )
    
    log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
    else:
        writer = None
    
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    clip_tokenizer = AutoTokenizer.from_pretrained(args.clip_path)
    
    new_token = ["<SEG>"]

    tokenizer.add_tokens(new_token)
    processor.tokenizer.add_tokens(new_token)
    seg_token_idx = processor.tokenizer("<SEG>", add_special_tokens=False).input_ids[0]

    model_kwargs = {
        "sam2_cfg": args.sam2_cfg,
        "precision": args.precision,
        "n_qformer_queries": args.n_qformer_queries,
        "n_qformer_layers": args.n_qformer_layers, 
        "adapter_dim": args.adapter_dim, 
        "seg_token_idx": seg_token_idx,
        "weight_dict": args.weight_dict,
        "reduction_ratio": args.reduction_ratio,
        "reduction_k": args.reduction_k,
        "_attn_implementation": "flash_attention_2",
    }
    qwen2_config = Qwen2VLConfig.from_pretrained(args.model_path)
    clip_config = CLIPTextConfig.from_pretrained(args.clip_path)
    config = LavidaConfig.from_qwen2vl_config(qwen2_config, clip_config, **model_kwargs)

    print("Initializing model...")
    model = LavidaForCausalLM.load_base_model(
        llm_ckpt=args.model_path,
        sam_ckpt=args.sam2_checkpoint,
        clip_ckpt=args.clip_path,
        config=config,
        attn_implementation="flash_attention_2", 
    )
        
    freeze_keywords = [
        "visual",
        "image_encoder", 
        "assist_clf"
    ]
    
    training_keywords = [
        "lm_head", 
        "embed_tokens",   
        "sam_mask_decoder", 
        "sam_prompt_encoder",
        "multi_scale_projector",
    ]

    for name, param in model.named_parameters():
        if any(keyword in name for keyword in freeze_keywords):
            param.requires_grad = False 
            
    lora_r = args.lora_r
    if lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in freeze_keywords + training_keywords 
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
    model.resize_token_embeddings(len(processor.tokenizer))
    
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in training_keywords
            ]
        ):
            p.requires_grad = True
            
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    
    sam2_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)), 
        transforms.ConvertImageDtype(torch.float32), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
    ])

    print("Loading datasets...")
    if not args.eval_only:
        train_dataset = HybridDataset(
            inference=False,
            datasets=args.train_sets,
            sampling_ratios=args.sampling_ratios,
            **args.dataset_args,
            list_root=args.list_root,
            total_sampled_frames=args.total_sampled_frames,
            frames_between_clips=args.frames_between_clip,
            llm_sample_frames=args.llm_sample_frames,
            augmentations=sam2_transform,
            anomaly_ratio=args.anomaly_ratio,
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=False,
            collate_fn=partial(collate_fn, clip_tokenizer=clip_tokenizer, processor=processor),
        )
        print(f"Training with {len(train_dataset)} examples.")
    else:
        train_dataset, train_loader = None, None
    
    if args.no_eval == False:
        val_dataset = HybridDataset(
            inference=True,
            datasets=args.test_sets,
            **args.dataset_args,
            list_root=args.list_root,
            total_sampled_frames=args.total_sampled_frames,
            frames_between_clips=args.frames_between_clip,
            llm_sample_frames=args.llm_sample_frames,
            augmentations=sam2_transform,
        )
        assert len(val_dataset.datasets) == 1
        val_sample_interval = val_dataset.datasets[0].sample_interval
        if args.test_sets[0] == ['xdviolence']:
            val_metric = "AP"
        else:
            val_metric = "AUC"
            
        assert args.val_batch_size == 1
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=True,  
            num_workers=args.workers,
            pin_memory=False,
            collate_fn=partial(collate_fn, clip_tokenizer=clip_tokenizer, processor=processor),
        )
        
        print(
            f"Validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        val_loader = None
        
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_base_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=args.epochs * args.steps_per_epoch,
    )
    
    if train_loader is not None and val_loader is not None:
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )
    elif train_loader is not None and val_loader is None:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )
    elif train_loader is None and val_loader is not None:
        model, optimizer, val_loader, scheduler = accelerator.prepare(
            model, optimizer, val_loader, scheduler
        )
    else:
        model, optimizer, scheduler = accelerator.prepare(
            model, optimizer, scheduler
        )
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        accelerator.load_state(args.resume)
        torch.cuda.empty_cache()

    
    if False:
        save_model = accelerator.unwrap_model(model).merge_and_unload()
        with open('model_structure.txt', 'w') as f:
            f.write(str(save_model))
        save_model.save_pretrained(
            args.ckpt_path,
            max_shard_size="4GB", 
            safe_serialization=True,
        )
        tokenizer.save_pretrained(args.ckpt_path)
        processor.save_pretrained(args.ckpt_path)
        exit()
        
    best_score= 0.
    f_roc, v_roc = 0.0, 0.0
    
    if args.eval_only:
        f_roc, v_roc = validate(val_loader, model, 0, writer,  args, val_sample_interval, val_metric)
        exit()
    
    train_iter = iter(train_loader)
    torch.cuda.empty_cache()
    for epoch in range(args.start_epoch, args.epochs):
        train_iter = train(
            accelerator,
            train_loader,
            model,
            epoch,
            optimizer,
            scheduler,
            writer,
            train_iter,
            args,
        )
        if args.no_eval == False:
            f_roc, v_roc = validate(val_loader, model, epoch, writer, args, val_sample_interval, val_metric)
            is_best = f_roc > best_score
            best_score = max(f_roc, best_score)
        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_base_dir, "ckpt_model")
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_base_dir,
                        "meta_log_froc{:.4f}_vroc{:.4f}.pth".format(
                            f_roc, v_roc
                        ),
                    ),
                )
                
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                accelerator.save_state(save_dir)
                print("save the best checkpoint.")

    dist.destroy_process_group()
    print("Finished!")
            

def train(
    accelerator,
    train_loader,
    model,
    epoch,
    optimizer,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    class_losses = AverageMeter("ClassLoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")
    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            class_losses,
            mask_dice_losses,
            mask_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        with accelerator.accumulate(model):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)
            video_name = input_dict.pop('video_path')
            data_time.update(time.time() - end)
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output_dict = model(**input_dict)
            loss = output_dict["total_loss"]
            ce_loss = output_dict["ce_loss"]
            class_loss = output_dict["loss_class"]
            mask_dice_loss = output_dict["loss_dice"]
            mask_loss = output_dict["loss_mask"]
            mask_iou_loss = output_dict["loss_iou"]
            accelerator.backward(loss)
            optimizer.step()
            
            size = len(input_dict["images"])
            losses.update(loss.item(), size)
            ce_losses.update(ce_loss.item(), size)
            class_losses.update(class_loss.item(), size)
            mask_dice_losses.update(mask_dice_loss.item(), size)
            mask_losses.update(mask_loss.item(), size)
            
        scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (global_step+1) % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()
                losses.all_reduce()
                ce_losses.all_reduce()
                class_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar("train/class_loss", class_losses.avg, global_step)
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )
            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            class_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter

@torch.inference_mode()
def validate(val_loader, model_engine, epoch, writer, args, sample_interval, val_metric):
    video_summery = VideoResultSummary(metric=val_metric)
    model_engine.eval()

    f_score, v_score = 0., 0.
    update_step = 0
    
    pred_masks = []
    target_masks = []

    with tqdm(val_loader, desc="Validating") as progress_bar:
        for input_dict in progress_bar:
            video_name = input_dict.pop('video_path')

            with torch.no_grad() and torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output_dict = model_engine(**input_dict, inference=True)

            if args.eval_pixel:
                p_mask = output_dict["pred_masks"].cpu()
                t_mask = output_dict["gt_masks"].unsqueeze(1).int().cpu()
                
                [H, W] = [158, 238]
                p_mask = F.interpolate(
                    p_mask.float(), 
                    size=(H, W),
                    mode="nearest", 
                )
                t_mask = F.interpolate(
                    t_mask.float(), 
                    size=(H, W),
                    mode="nearest", 
                ).int()
                pred_masks.append(p_mask)
                target_masks.append(t_mask)
            
            pred_score = output_dict["pred_scores"].view(1, -1)# .repeat(1, sample_interval)
            pred_score = torch.repeat_interleave(pred_score, sample_interval, dim=1)
            gt_label = output_dict["gt_labels"].int().view(1, -1)[:, :pred_score.size(1)]
            assert len(video_name) == 1

            video_summery.update(
                video_names=video_name, 
                pred_scores=pred_score.to(torch.float32), 
                gt_labels=gt_label,
            )
            update_step += 1
            if update_step % 1 == 0:
                f_score, v_score = video_summery.calculate_score()
                progress_bar.set_postfix(f_score=f"{f_score:.4f}", v_score=f"{v_score:.4f}")

    frame_avg_score, video_avg_score = video_summery.calculate_score()
    if args.eval_pixel:
        pred_masks = torch.cat(pred_masks, dim=0).flatten().numpy()
        target_masks = torch.cat(target_masks, dim=0).flatten().numpy()
        
        pixel_auc = roc_auc_score(target_masks, pred_masks)
        print(f"Pixel Score: {pixel_auc:.4f}")
    
    if args.local_rank == 0:
        writer.add_scalar(f"val/frame_avg", frame_avg_score, epoch)
        writer.add_scalar(f"val/video_avg", video_avg_score, epoch)
        print(f"Frame Level Score: {frame_avg_score:.4f}, Video Level Score: {video_avg_score:.4f}")
        

    return frame_avg_score, video_avg_score


if __name__ == "__main__":
    args = load_args(args_file='./train/train_para.yaml')
    main(args)