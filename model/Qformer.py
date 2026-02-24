import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from sam2.modeling.sam.transformer import Attention, MLP, TwoWayAttentionBlock


class TwoWayQformer(nn.Module):
    def __init__(self, 
                 d_model: int = 768,
                 output_dim: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 12,
                 n_queries: int = 32,
                 d_ff: int = 3072,
                 dropout: float = 0.1):
        super().__init__()
        
        self.output_dim = output_dim
        self.d_model = d_model
        self.n_queries = n_queries
        
        self.query_tokens = nn.Parameter(torch.randn(n_queries, d_model))
        self.transformer_layers = nn.ModuleList([
            TwoWayAttentionBlock(
                    embedding_dim=d_model,
                    num_heads=n_heads,
                    mlp_dim=d_ff,
                    attention_downsample_rate=2,
                    skip_first_layer_pe=False,
                )
            for _ in range(n_layers)
        ])
        
        self.output_q_proj = MLP(input_dim=d_model*n_queries, hidden_dim=d_model, output_dim=output_dim, num_layers=1)
        self.output_k_proj = MLP(input_dim=d_model, hidden_dim=d_model, output_dim=output_dim, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        exp_q = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        queries = exp_q
        keys = x
        for layer in self.transformer_layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=exp_q,
                key_pe=x,
            )
        queries = queries.view(batch_size, -1) # (batch_size, n_queries * output_dim)
        output_q = self.output_q_proj(queries).view(batch_size, -1, self.output_dim)
        output_k = self.output_k_proj(keys)
        return output_q, output_k
        