from typing import List
from collections import OrderedDict
import os
from typing import List, Any, Callable, Optional, Union, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLModel, CLIPConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
import torchvision.transforms.v2 as transforms
from hydra import compose, initialize
from omegaconf import OmegaConf
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from transformers import logging as transformers_logging
import logging

from .model import LavidaMetaModel
from .Config import LavidaConfig
from .ClipClf import ClipClassifier
from .build_model import build_model
from loss import MultiStepMultiMasksAndIous

pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)


class LavidaForCausalLM(Qwen2VLForConditionalGeneration):

    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config)
        self.seg_token_idx = config.seg_token_idx
        self.sam2_cfg = config.sam2_cfg
        self.precision = config.precision 
        self.reduction_ratio = config.reduction_ratio
        self.reduction_k = config.reduction_k
        kwargs["llm_feature_dim"] = config.hidden_size
        kwargs["n_qformer_layers"] = config.n_qformer_layers
        kwargs["n_qformer_queries"] = config.n_qformer_queries
        kwargs["adapter_dim"] = config.adapter_dim
        weight_dict = config.weight_dict 
        if self.precision == 'float16' or self.precision == 'fp16':
            self.precision = torch.float16
        elif self.precision == 'float32' or self.precision == 'fp32':
            self.precision = torch.float32
        elif self.precision == 'bfloat16' or self.precision == 'bf16':
            self.precision = torch.bfloat16
        
        if weight_dict:
            self.seg_loss = MultiStepMultiMasksAndIous(
                weight_dict=weight_dict,
                iou_use_l1_loss=True,
                pred_obj_scores=True,
                focal_gamma_obj_score=0.0,
                focal_alpha_obj_score=-1.0,
            )

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
         
        self.post_init()
        self.seg_model = build_model(
            config_file=self.sam2_cfg, 
            sam2_checkpoint=None, 
            kwargs=kwargs, 
            load_weights=False,
        )
        
        self.assist_clf = ClipClassifier(
            config=CLIPConfig(**config.clip_config),
        )
        
        
    @classmethod
    def load_base_model(
        cls,
        llm_ckpt: str,
        sam_ckpt: str,
        clip_ckpt: str,
        config: LavidaConfig,
        **kwargs
    ):
        if config.precision == 'float16' or config.precision == 'fp16':
            torch_dtype = torch.float16
        elif config.precision == 'float32' or config.precision == 'fp32':
            torch_dtype = torch.float32
        elif config.precision == 'bfloat16' or config.precision == 'bf16':
            torch_dtype = torch.bfloat16
        
        original_level = {
            'root': logging.getLogger().level,
            'transformers': transformers_logging.get_verbosity()
        }
        
        logging.disable(logging.CRITICAL)  
        transformers_logging.set_verbosity_error() 
        
        lavida_model = super().from_pretrained(
            llm_ckpt,
            output_loading_info=False,
            device_map=None,  
            low_cpu_mem_usage=False, 
            torch_dtype=torch_dtype,
            config=config,
        )
        logging.getLogger().setLevel(original_level['root'])
        transformers_logging.set_verbosity(original_level['transformers'])

        sam_state = torch.load(sam_ckpt, map_location="cpu", weights_only=True)
        if "model" in sam_state: 
            sam_state = sam_state["model"]
        lavida_model.seg_model.load_state_dict(sam_state, strict=False)

        lavida_model.assist_clf = ClipClassifier.load_base_model(
            clip_ckpt,
            device_map=None,  
            low_cpu_mem_usage=False, 
            torch_dtype=torch_dtype,
            attn_implementation=kwargs.get("attn_implementation", "flash_attention_2"),
        )
        return lavida_model
    
    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(  
        self,  
        input_ids: torch.LongTensor, 
        images: List[torch.FloatTensor],
        pixel_values: List[torch.FloatTensor],
        pixel_values_videos: List[torch.FloatTensor], 
        frame_idx: List[int],
        anomaly_types: List[List[str]],
        attention_mask: torch.LongTensor,  
        image_grid_thw: torch.LongTensor,
        video_grid_thw: torch.LongTensor,  
        labels: torch.LongTensor,  
        anomaly_labels: List[torch.LongTensor],
        target_masks: torch.LongTensor,  
        target_labels: torch.LongTensor,
        inference: bool = False,  
        use_cache: Optional[bool] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,  
    ):  
        '''
         Args:
            input_ids: Tokenized text input [batch_size, seq_len, hidden_size]
            images: List of image tensors for segmentation model 
                   [batch_size][time_steps, channels, height, width]
            pixel_values: Processed image inputs for base model 
                         [batch_size][channels, height, width] 
            pixel_values_videos: Processed video inputs for base model
                                [batch_size][time_steps, channels, height, width]
            frame_idx: Frame indices within original video [batch_size]
            anomaly_types: Text descriptions of anomaly types [batch_size][num_types]
            attention_mask: Attention mask for text tokens [batch_size, seq_len]
            image_grid_thw: Spatial-temporal grid info for images [specific_shape]
            video_grid_thw: Spatial-temporal grid info for videos [specific_shape]
            labels: Shifted text labels for language modeling [batch_size, seq_len]
            anomaly_labels: Ground truth anomaly labels [batch_size]
            target_masks: Ground truth segmentation masks 
                         [batch_size, time_steps, channels, height, width]
            target_labels: Ground truth segmentation labels [batch_size, time_steps]
            inference: If True, returns predictions; if False, computes losses
            **kwargs: Additional forward arguments
        
        Returns:
            During inference:
                Dict containing:
                - pred_masks: Predicted segmentation masks [batch*time_steps, num_masks, height, width]
                - pred_ious: Predicted IoU scores [batch*time_steps, num_masks]
                - pred_scores: Objectness scores [batch*time_steps, 1]
                - gt_masks: Ground truth masks (None if not provided in input)
                - gt_labels: Ground truth labels (None if not provided in input)

            During training:
                Dict containing:
                - total_loss: Combined segmentation + language modeling loss
                - ce_loss: Cross-entropy language modeling loss
                - [other segmentation losses from seg_loss]
        '''

        output, input_ids = self.llm_forward(
            input_ids=input_ids, 
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            labels=labels,
            output_hidden_states=True,
            use_cache=use_cache,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            inference=inference,
        )
        output_hidden_states = output.hidden_states
        ce_loss = output.loss
        
        seg_token_mask = input_ids == self.seg_token_idx
        seg_features = [h[m.to(h.device)] for h, m in zip(output_hidden_states[-1], seg_token_mask)] # B * [n_seg_tokens, D]
        assert len(seg_features) > 0

        cls_features = []
        for i in range(len(anomaly_types)):
            cls_features.append(self.assist_clf.encode_text(anomaly_types[i]))
            

        seg_output, cls_output = self.seg_model(
            images=images,
            frame_idx=frame_idx,
            seg_prompts=seg_features,  
            class_prompts=cls_features, 
            inference=inference,
        )
         
        images_length = [img.shape[0] for img in images]
        
        (
            pred_masks,                 # [B*T, M, H, W]
            ious,                       # [B*T, M]
            object_score_logits,        # [B*T, 1] 
        ) = self.seg_model.process_output(
            seg_output,
            images_length=images_length,
            output_multimask=not inference,
            split_by_object=True,
            split_by_lengths=False,
        )[0]
        
        if inference:
            return {
                'pred_masks': pred_masks,
                'pred_ious': ious,
                'pred_scores': object_score_logits, 
                'gt_masks': target_masks,
                'gt_labels': target_labels,
            }
        
        loss_dict = self.seg_loss(
            pred_masks=pred_masks,  #  [B*T, M, H, W]
            pred_ious=ious,          # [B*T, M]
            object_score_logits=object_score_logits,  # [B*T]
            targets=target_masks,#.flatten(0, 1),       # [B, T, H, W] -> [B*T, H, W]
            target_obj=target_labels,#.flatten(0, 1),     # [B*T]
        )   
        loss_dict["total_loss"] = loss_dict["total_loss"]
        
        loss_dict["ce_loss"] = ce_loss
        loss_dict["total_loss"] = loss_dict["total_loss"] + ce_loss
        return loss_dict

    def llm_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,   
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,  
        cache_position: Optional[torch.LongTensor] = None,
        inference: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
        elif inputs_embeds is not None:
            batch_size, seq_len, _ = inputs_embeds.shape
            device = inputs_embeds.device
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                if inference:
                    video_embeds, input_ids, inputs_embeds, attention_mask, labels = self.video_embed(
                        input_ids=input_ids, input_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, hidden_states=pixel_values_videos, grid_thw=video_grid_thw
                    )
                else:
                    video_embeds = self.visual(pixel_values_videos, video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if self.reduction_ratio < 1:
            with torch.no_grad():
                if labels is None:
                    labels = input_ids.clone()
                input_ids, inputs_embeds, attention_mask, labels = self.token_reduction(
                    input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels
                )

        outputs = super().forward(
            input_ids=input_ids if inputs_embeds is None else None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            **kwargs,
        )
        return outputs, input_ids
    
    
    def video_embed(
        self, 
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        hidden_states: torch.Tensor, 
        grid_thw: torch.Tensor
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        hidden_states = self.visual.patch_embed(hidden_states)
        rotary_pos_emb = self.visual.rot_pos_emb(grid_thw)
        with torch.no_grad():
            reduction_mask, seqlens = self.temporal_token_reduction(hidden_states, grid_thw)
        
        cu_seqlens = seqlens.cumsum(dim=0)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(torch.int32)
        hidden_states = hidden_states[reduction_mask.bool()]
        rotary_pos_emb = rotary_pos_emb[reduction_mask.bool()]
        for blk in self.visual.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
            
        frame_counts = grid_thw[:, 0]  # [B]
        seqlens_splits = torch.split(seqlens, frame_counts.tolist())  
        keep_video_token_num = torch.tensor([split.sum().item()/4 for split in seqlens_splits], device=device, dtype=torch.int32)  # [B]
        
        video_token_mask = input_ids == self.config.video_token_id  # [B, L]
        pad_mask = input_ids == self.config.bos_token_id  # [B, L]
        video_token_counts = video_token_mask.sum(dim=1)  # [B]
        keep_mask = ~pad_mask  # [B, L]
        
        has_video_tokens = video_token_counts > 0  # [B]
        sample_indices = has_video_tokens.nonzero(as_tuple=True)[0]  
        
        for i, sample_idx in enumerate(sample_indices):
            keep_num = keep_video_token_num[i]  
            
            sample_video_mask = video_token_mask[sample_idx]  # [L]
            video_token_indices = sample_video_mask.nonzero(as_tuple=True)[0]  # [video_token_count]
            if len(video_token_indices) > keep_num:
                remove_indices = video_token_indices[keep_num:]
                keep_mask[sample_idx, remove_indices] = False
        new_input_ids = []
        new_attention_mask = []
        new_input_embeds = []
        new_labels = []
        if labels is None:
            labels = input_ids.clone()
        
        for i in range(batch_size):
            mask = keep_mask[i]
            new_input_ids.append(input_ids[i][mask])
            new_attention_mask.append(attention_mask[i][mask])
            new_input_embeds.append(input_embeds[i][mask])
            new_labels.append(labels[i][mask])
        
        new_input_ids, new_input_embeds, new_attention_mask, new_labels = self.pad_batch(
            new_input_ids, new_input_embeds, new_attention_mask, new_labels
        )
        return self.visual.merger(hidden_states), new_input_ids, new_input_embeds, new_attention_mask, new_labels
    
    def temporal_token_reduction(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = grid_thw.shape[0]
        device = hidden_states.device
        
        token_nums = grid_thw.prod(dim=1)  # [B]
        split_indices = token_nums.tolist()
        batch_hidden_states = torch.split(hidden_states, split_indices, dim=0)
        
        all_masks = []
        all_frame_token_counts = []
        
        for i in range(batch_size):
            t, h, w = grid_thw[i].tolist()
            sample_tokens = batch_hidden_states[i]  # [t*h*w, 1280]
            sample_3d = sample_tokens.reshape(t, h*w//4, -1)  # [t, h*w, 1280]
            
            if t > 1:
                first_frame = sample_3d[0:1]  # [1, h*w, 1280]
                first_frame_expanded = first_frame.expand(t, h*w//4, -1)  # [t, h*w, 1280]
                cosine_sim = F.cosine_similarity(sample_3d, first_frame_expanded, dim=-1)  # [t, h*w]
                frame_diffs = 1 - cosine_sim  # [t, h*w]
                
                avg_frame_diff = frame_diffs[1:].mean(dim=0)  # [h*w]
                threshold = 0.01
                
                mask = torch.ones(t, h*w//4, device=device)
                static_regions = avg_frame_diff < threshold  # [h*w]
                mask[1:, static_regions] = 0
                mask = torch.repeat_interleave(mask, 4, dim=1)
                mask = mask.reshape(t, h, w)  # [t, h, w]
            else:
                mask = torch.ones(t, h, w, device=device)
            
            mask_flat = mask.flatten()  # [t*h*w]
            all_masks.append(mask_flat)
            
            frame_token_counts = mask.reshape(t, -1).sum(dim=1).int()  # [t]
            all_frame_token_counts.append(frame_token_counts)
            
        final_mask = torch.cat(all_masks, dim=0)  # [L]
        frame_token_counts = torch.cat(all_frame_token_counts, dim=0) 
        
        return final_mask, frame_token_counts
    
    
    @torch.no_grad()
    def visual_token_sampling(self, visual_embds, reduction_ratio, k):
        L, C = visual_embds.shape
        visual_embds_float = visual_embds.to(torch.float32)
        dist_matrix = torch.cdist(visual_embds_float, visual_embds_float) / (C ** 0.5)  # [seq_len, seq_len]
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k+1, dim=-1, largest=False)
        k_distances = dist_nearest[:, 1:]  # [seq_len, k]
        neighbor_indices = index_nearest[:, 1:]  # [seq_len, k]
        k_distance_values = k_distances[:, -1]   # [seq_len]
        k_distance_expanded = k_distance_values.unsqueeze(0).expand(L, L)  # [seq_len, seq_len]
        reach_dist = torch.max(k_distance_expanded, dist_matrix)
        reach_dist_neighbors = torch.gather(reach_dist, dim=1, index=neighbor_indices)  # [seq_len, k]
        # LRD(p) = k / Σ(reach_dist(p,o))
        lrd = k / (reach_dist_neighbors.sum(dim=1) + 1e-10)  # [seq_len] 
        R = max(1, int(L * reduction_ratio))
        _, lrd_topk_indices = torch.topk(lrd, k=R, dim=0, largest=True)
        cluster_centers_idx, _ = torch.sort(lrd_topk_indices)  # [R]
        cluster_centers = visual_embds_float[cluster_centers_idx]  # [R, C]
        dist_to_centers = torch.cdist(visual_embds_float, cluster_centers.float()) / (C ** 0.5)  # [L, R]
        assignments = torch.argmin(dist_to_centers, dim=1)  # [L]
        
        one_hot = torch.zeros(L, R, device=visual_embds.device)
        one_hot.scatter_(1, assignments.unsqueeze(1), 1)  # [L, R]
        
        mask = one_hot.t().bool()  # [R, L]

        attention_scores = cluster_centers @ visual_embds_float.t() / (C ** 0.5)  # [R, L]
        attention_scores = -attention_scores
        attention_scores = attention_scores.masked_fill(~mask, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=1)  # [R, L]
        merged_features = attention_weights @ visual_embds_float  # [R, C]

        return merged_features.to(visual_embds.dtype), cluster_centers_idx

    
    def token_reduction(self, input_ids, inputs_embeds, attention_mask, labels):
        batch_size = input_ids.shape[0]
        results = []
        input_ids_list, inputs_embeds_list, attention_mask_list, labels_list = [], [], [], []
        for i in range(batch_size):
            visual_mask = (input_ids[i] == self.config.image_token_id) | \
                        (input_ids[i] == self.config.video_token_id)   # [seq_len]
            visual_indices = torch.where(visual_mask)[0]
            
            if len(visual_indices) > 0:
                visual_embds = inputs_embeds[i][visual_indices]
                
                merged_features, cluster_centers_idx = self.visual_token_sampling(visual_embds, self.reduction_ratio, k=self.reduction_k)
                keep_mask = torch.ones(len(input_ids[i]), dtype=torch.bool, device=input_ids.device)
                keep_mask[visual_indices] = False
                selected_visual_indices = visual_indices[cluster_centers_idx] 
                keep_mask[selected_visual_indices] = True
                inputs_embeds[i][selected_visual_indices] = merged_features
                
            else:
                keep_mask = torch.ones(len(input_ids[i]), dtype=torch.bool, device=input_ids.device)
            input_ids_list.append(input_ids[i][keep_mask])
            inputs_embeds_list.append(inputs_embeds[i][keep_mask])
            attention_mask_list.append(attention_mask[i][keep_mask])
            labels_list.append(labels[i][keep_mask])
        
        new_input_ids, new_inputs_embeds, new_attention_mask, new_labels = self.pad_batch(
            input_ids_list, inputs_embeds_list, attention_mask_list, labels_list
        )
        return new_input_ids, new_inputs_embeds, new_attention_mask, new_labels
    
    def pad_batch(self, input_ids_list, inputs_embeds_list, attention_mask_list, labels_list):
        max_len = max(r.shape[0] for r in input_ids_list)
        device = input_ids_list[0].device
        batch_size = len(input_ids_list)
        
        new_input_ids = torch.full((batch_size, max_len), self.config.bos_token_id, dtype=input_ids_list[0].dtype, device=device)
        new_inputs_embeds = self.model.embed_tokens(new_input_ids)
        new_attention_mask = torch.zeros(batch_size, max_len, dtype=attention_mask_list[0].dtype, device=device)
        new_labels = torch.full((batch_size, max_len), -100, dtype=labels_list[0].dtype, device=device)
        
        for i in range(batch_size):
            seq_len = input_ids_list[i].shape[0]
            new_input_ids[i, -seq_len:] = input_ids_list[i]
            new_inputs_embeds[i, -seq_len:] = inputs_embeds_list[i]
            new_attention_mask[i, -seq_len:] = attention_mask_list[i]
            new_labels[i, -seq_len:] = labels_list[i]
        return new_input_ids, new_inputs_embeds, new_attention_mask, new_labels
        
