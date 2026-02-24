from hydra import compose, initialize
from omegaconf import OmegaConf
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2_video_predictor
import torch


def build_model(
    config_file, 
    sam2_checkpoint, 
    kwargs, 
    apply_postprocessing=True, 
    hydra_overrides_extra=[],
    load_weights=True,
):
    GlobalHydra.instance().clear()
    hydra_overrides = [
        "++model._target_=model.LavidaMetaModel",  
         *(f"++model.{key}={value}" for key, value in kwargs.items())  
    ]
    
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    
    hydra_overrides.extend(hydra_overrides_extra)
    
    with initialize(config_path="../sam2"):
        cfg = compose(config_name=config_file, overrides=hydra_overrides)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    
    if load_weights and sam2_checkpoint:
        sam_model = build_sam2_video_predictor(config_file, sam2_checkpoint, device='cuda')
        model.load_state_dict(sam_model.state_dict(), strict=False)
        model = model.to(device)

    return model