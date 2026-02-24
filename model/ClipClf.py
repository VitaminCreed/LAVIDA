import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoTokenizer, CLIPModel, CLIPConfig, CLIPTextModel, CLIPTextConfig
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
from torch.nn.utils.rnn import pad_sequence

class ClipClassifier(CLIPTextModel):
    def __init__(
        self,
        config: CLIPTextConfig,
    ):
        super().__init__(config)
        self.post_init()

    
    @classmethod
    def load_base_model(cls, model_path, **kwargs):
        model = super().from_pretrained(model_path, **kwargs)
        return model
    
    @property
    def text_embedding_dim(self):
        return self.text_model.embeddings.token_embedding.weight.shape[1]
    
    # @property
    # def patch_size(self):
    #     return self.vision_model.embeddings.patch_size
    

    def encode_text(self, inputs: torch.Tensor, device: torch.device = None):
        outputs = self.text_model(**inputs.to(self.device))
        text_features = outputs.pooler_output
        return text_features
    



