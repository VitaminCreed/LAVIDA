
from transformers import Qwen2VLConfig 

class LavidaConfig(Qwen2VLConfig):
    model_type = "lavida"  
    
    def __init__(
        self,
        sam2_cfg=None,
        precision=None,
        seg_token_idx=None,
        weight_dict=None,
        clip_config=None,
        n_qformer_queries=None,
        n_qformer_layers=None,
        adapter_dim=None,
        reduction_ratio=None,
        reduction_k=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.sam2_cfg = sam2_cfg or {}
        self.precision = precision
        self.seg_token_idx = seg_token_idx
        self.weight_dict = weight_dict or {}
        self.clip_config = clip_config or {}
        self.n_qformer_queries = n_qformer_queries
        self.n_qformer_layers = n_qformer_layers
        self.adapter_dim = adapter_dim
        self.reduction_ratio = reduction_ratio
        self.reduction_k = reduction_k
    
    @classmethod
    def from_qwen2vl_config(cls, qwen2vl_config, clip_config, **add_kwargs):
        base_config = qwen2vl_config.to_dict()
        base_config.pop("_name_or_path", None)
        base_config.pop("torch_dtype", None)
        
        clip_config = clip_config.to_dict()
        base_config.update({"clip_config": clip_config})

        base_config.update(add_kwargs)
        return cls(**base_config)
    
    def to_dict(self):
        output = super().to_dict()
        output.update({
            "sam2_cfg": self.sam2_cfg,
            "precision": self.precision,
            "seg_token_idx": self.seg_token_idx,
            "weight_dict": self.weight_dict,
        })
        
        return output
    
