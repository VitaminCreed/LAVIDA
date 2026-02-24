from collections import OrderedDict
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from torchvision import transforms
from decord import VideoReader, cpu

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.modeling.sam.transformer import Attention, MLP
from .SAM2AnomalyDetector import SAM2AnomalyDetector
from .Qformer import TwoWayQformer

from collections import OrderedDict
from typing import Any, List, Optional, Tuple, Type


class LavidaMetaModel(SAM2AnomalyDetector):
    def __init__(
        self,
        llm_feature_dim: int,
        n_qformer_layers: int,
        n_qformer_queries: int,
        adapter_dim: int,
        **kwargs,
    ):
        super(LavidaMetaModel, self).__init__(**kwargs)
        visual_dims = [32, 64, 256]

        self.multi_scale_projector = MultiScaleSemanticProjector(
            sam_prompt_embed_dim=self.sam_prompt_embed_dim,
            llm_feature_dim=llm_feature_dim,
            n_qformer_layers=n_qformer_layers,
            n_qformer_queries=n_qformer_queries,
            n_heads=12,
            d_ff=2048,
            text_dim=512,
            vision_dim=256,
            hidden_dim=adapter_dim,
            dropout=0.1,
        )
        self.set_rope_attention_dropout_to_zero()
        
    def set_rope_attention_dropout_to_zero(self):
        import warnings
        from sam2.modeling.sam.transformer import RoPEAttention
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            device_props = torch.cuda.get_device_properties(device_id)
            compute_capability = device_props.major * 10 + device_props.minor

            if compute_capability in [86, 89]:
                print(f'''The GPU architecture is sm{compute_capability}. Flash attention currently doesn't support training with head_dim ∈ (192, 224] or (head_dim ∈ (224, 256] and dropout > 0.0) on gpu architectures in the range[sm86, sm89]. Here dropout has been set to 0. in attention modules.''')
            for name, module in self.named_modules():
                if isinstance(module, RoPEAttention): 
                    module.dropout_p = 0.0
        else:
            print("CUDA is not available.")    
    
    def process_output(self, sam_output, images_length=[], split_by_object=True, split_by_lengths=False, output_multimask=True):
        '''
        Align the output of SAM2 with the labels in `batch_data`.
        Args:
            sam_output (List[Dict]): The output of SAM2. Length: Total frames or images.
                pred_masks_high_res: [n_obj, num_mask, H, W]
                ious: [n_obj, num_mask]
                object_score_logits: [n_obj, 1]
            images_length: List[int]: The length of each video or image.
        Returns:
            result (List[List]): Length: n_obj, in video anomaly detection, n_obj = 1.
                pred_masks: [Total_images, num_mask, H, W]
                ious: [Total_images, num_mask]
                object_score_logits: [Total_images, 1]
        '''
        pred_masks, ious, object_score_logits= [], [], []
        for p in sam_output:
            pred_masks.append(p['pred_masks_high_res'].unsqueeze(1))
            if isinstance(p['ious'], tuple):
                p['ious'] = p['ious'][0]
            ious.append(p['ious'].unsqueeze(1) if p['ious'] is not None else torch.zeros(1, 1))
            object_score_logits.append(p['object_score_logits'])
        
        pred_masks = torch.cat(pred_masks, dim=1)       # [n_obj, B*T, num_mask, H, W]
        ious = torch.cat(ious, dim=1)                   # [n_obj, B*T]
        object_score_logits = torch.cat(object_score_logits, dim=1)  # [n_obj, B*T]
        result = (pred_masks, ious, object_score_logits)
        
        num_masks = 3 if output_multimask else 1
        assert num_masks == pred_masks.shape[2]
       
        if split_by_object:
            n_obj, _, _, H, W = pred_masks.shape
            pred_masks = torch.unbind(pred_masks.reshape(n_obj, -1, num_masks, H, W), dim=0) # n_obj*[B*T, M, H, W]
            ious = torch.unbind(ious.reshape(n_obj, -1, num_masks), dim=0)
            object_score_logits = torch.unbind(object_score_logits.reshape(n_obj, -1), dim=0)
            
        if split_by_lengths and images_length:
            pred_masks = [torch.split(m, images_length, dim=0) for m in pred_masks]
            ious = [torch.split(i, images_length, dim=0) for i in ious]
            object_score_logits = [torch.split(o, images_length, dim=0) for o in object_score_logits]
            
        result = list(zip(pred_masks, ious, object_score_logits))
        return result
    
    
    def expand_feature(self, vision_feats, vision_pos_embeds, expand_dim):
        """
        Expand vision features and position embeddings to num_obj, for multi-object segmentation in SAM2.
        """
        expanded_vision_feats = vision_feats.copy()
        expanded_vision_pos_embeds = vision_pos_embeds.copy()
        
        for i, feat in enumerate(expanded_vision_feats):
            exband_feat = feat.expand(-1, expand_dim, -1)
            expanded_vision_feats[i] = exband_feat # + visual_adapted
        
        for i, pos_embed in enumerate(expanded_vision_pos_embeds):
            expanded_vision_pos_embeds[i] = pos_embed.expand(-1, expand_dim, -1)
        return expanded_vision_feats, expanded_vision_pos_embeds
 
    
    def forward(
        self, 
        images: List[torch.Tensor],
        seg_prompts: List[torch.Tensor],
        class_prompts: torch.Tensor,
        frame_idx: List[dict],
        inference: bool = False,
    ):
        seg_output = []
        cls_output = []
            
        for i in range(len(images)):
            if frame_idx[i] is None:
                out, cls_out = self.seg_image(images[i], seg_prompts[i], class_prompts[i], inference)
            else:
                out, cls_out = self.seg_video(images[i], seg_prompts[i], class_prompts[i], frame_idx[i],inference)
            seg_output.extend(out)
            cls_output.append(cls_out)
        return seg_output, cls_output
    
    def seg_image(
        self, 
        images: torch.Tensor,  # [B, C, H, W]
        seg_prompts: torch.Tensor,  # [B, 1, sam2_dim]
        class_prompts: torch.Tensor,
        inference: bool = False,
    ):
        self.training = not inference
        backbone_out = self.forward_image(images)
        (
            _,
            vision_feats,    # List([65536, n_frame, 32], [16384, n_frame, 64], [4096, n_frame, 256])
            vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features(backbone_out)
        
        seg_prompts, cls_output = self.multi_scale_projector(
            seg_emb=seg_prompts, 
            text_feat=class_prompts, 
            vision_feat=vision_feats[-1], 
            vision_pos= vision_pos_embeds[-1],
        )
        
        feature = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(vision_feats, feat_sizes)
        ]
        output_dict = {}
        if len(vision_feats) > 1:
            pix_feat, high_res_features = feature[-1], feature[:-1]
        else:
            pix_feat = feature
            high_res_features = None
        multimask_output = not inference
        sam_outputs = self._forward_sam_heads(
            backbone_features=pix_feat,
            point_inputs=None,
            mask_inputs=None, 
            seg_feature_inputs=seg_prompts,
            high_res_features=high_res_features,
            multimask_output=multimask_output,
        )   
        
        (
            multi_iou,
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs
        
        if multimask_output:
            output_dict["pred_masks_high_res"] = high_res_multimasks
            output_dict["ious"] = multi_iou,
        else:
            output_dict["pred_masks_high_res"] = high_res_masks
            output_dict["ious"] = ious

        output_dict["obj_ptr"] = obj_ptr
        
        output_dict["object_score_logits"] = object_score_logits
        
        return [output_dict], cls_output

        
    def seg_video(
        self, 
        images: torch.Tensor,  # [T, C, H, W] or [1, C, H, W]
        seg_prompts: torch.Tensor,  # [1, 1, sam2_dim]
        class_prompts: torch.Tensor,  # [n_cls, 1, sam2_dim]
        frame_idx: dict | None,  # [T]
        inference: bool = False,
    ):
        self.training = not inference
        output_dict = {
            "cond_frame_outputs": {}, 
            "non_cond_frame_outputs": {}, 
        }
        
        cond_frame_idx = np.array(frame_idx["cond_frame_idx"])
        non_cond_frame_idx = np.array(frame_idx["non_cond_frame_idx"])
        all_frame_idx = frame_idx["all_frame_idx"]
        
        n_frame = images.shape[0]
        backbone_out = self.forward_image(images)  # 输入维度 [T, C, H, W]
        
        output_multimasks = not inference
        
        (
            _,
            vision_feats,    # List([65536, n_frame, 32], [16384, n_frame, 64], [4096, n_frame, 256])
            vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features(backbone_out)
        
        seg_prompts, cls_output = self.multi_scale_projector(
            seg_emb=seg_prompts, 
            text_feat=class_prompts, 
            vision_feat=vision_feats[-1], 
            vision_pos= vision_pos_embeds[-1],
        )

        # seg_prompts = seg_prompts.expand(-1, n_frame, -1)  # [1, n_frame, sam2_dim]
        

        for idx in cond_frame_idx:
            current_vision_feats = [x[:, idx:idx+1, :] for x in vision_feats]
            current_vision_pos_embeds = [x[:, idx:idx+1, :] for x in vision_pos_embeds]
            
            if seg_prompts.shape[0] > 1:
                current_vision_feats, current_vision_pos_embeds = self.expand_feature(current_vision_feats, current_vision_pos_embeds, seg_prompts.shape[0])
            
            current_out = self.track_step(
                frame_idx=idx,
                is_init_cond_frame=True,
                current_vision_feats=current_vision_feats,      # List([65536, num_class, 32], [16384, num_class, 64], [4096, num_class, 256])
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=None,
                mask_inputs=None, 
                seg_feature_inputs=seg_prompts[:, idx:idx+1, :],                # [num_class, 1, sam2_dim], num_class=1
                output_dict=output_dict,
                num_frames=n_frame,
                output_multimasks=output_multimasks,
            )
            output_dict["cond_frame_outputs"][idx] = current_out
            
        for idx in non_cond_frame_idx:
            current_vision_feats = [x[:, idx:idx+1, :] for x in vision_feats]
            current_vision_pos_embeds = [x[:, idx:idx+1, :] for x in vision_pos_embeds]
            
            if seg_prompts.shape[0] > 1:
                current_vision_feats, current_vision_pos_embeds = self.expand_feature(current_vision_feats, current_vision_pos_embeds, seg_prompts.shape[0])
            
            current_out = self.track_step(
                frame_idx=idx,
                is_init_cond_frame=True,
                current_vision_feats=current_vision_feats,      # [(H*W), num_class, C]
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=None,
                mask_inputs=None, 
                seg_feature_inputs=seg_prompts[:, idx:idx+1, :], # [:, nearest_cond_idx:nearest_cond_idx+1, :],                # [num_class, clip_dim]
                class_feature_inputs=None, # class_prompts
                output_dict=output_dict,
                num_frames=n_frame,
                output_multimasks=output_multimasks,
            )
            output_dict["non_cond_frame_outputs"][idx] = current_out
        
        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
        all_frame_outputs = [all_frame_outputs[t] for t in all_frame_idx]
        # Make DDP happy with activation checkpointing by removing unused keys
        all_frame_outputs = [
            {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
        ]
        return all_frame_outputs, cls_output
    

    
class MultiScaleSemanticProjector(nn.Module):
    def __init__(
        self,
        sam_prompt_embed_dim: int,
        llm_feature_dim: int,
        n_qformer_layers: int,
        n_qformer_queries: int,
        n_heads: int,
        d_ff: int,
        text_dim: int,
        vision_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        
        # Projection layers
        self.video_proj = nn.Linear(vision_dim, hidden_dim)  # [vision_dim → hidden_dim]
        self.text_proj = nn.Linear(text_dim, hidden_dim)     # [text_dim → hidden_dim]
        
        # Cross-attention 
        self.cross_attn = Attention(hidden_dim, num_heads=8, downsample_rate=2)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)  # [hidden_dim → hidden_dim]
        
        # Q-Former for projection
        self.seg_proj = nn.Linear(llm_feature_dim, hidden_dim)
        self.text_fc = TwoWayQformer(
            output_dim=sam_prompt_embed_dim,
            d_model=hidden_dim,
            n_heads=n_heads,
            n_layers=n_qformer_layers,
            n_queries=n_qformer_queries,
            d_ff=d_ff,
            dropout=dropout,
        )
        

    def forward(
        self,
        seg_emb: torch.Tensor,
        text_feat: torch.Tensor,
        vision_feat: torch.Tensor,
        vision_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process text and vision features to generate segmentation prompts and class embeddings.
        
        Args:
            text_feat: Text features of shape [n, text_dim]
            vision_feat: Vision features of shape [L, t, vision_dim]
            vision_pos: Positional encoding for vision features of shape [L, t, vision_dim]
            
        Returns:
            seg_prompts: Segmentation prompts of shape [1, t, sam_prompt_embed_dim]
            cls_emb: Class embeddings of shape [t, n_cls, sam_prompt_embed_dim]
        """
        t = vision_feat.size(1)
        
        # Project features to hidden dimension
        text_proj = self.text_proj(text_feat)  # [n, text_dim] → [n, hidden_dim]
        video_proj = self.video_proj(vision_feat + vision_pos)  # [L, t, vision_dim] → [L, t, hidden_dim]
        video_proj = video_proj.permute(1, 0, 2)  # [t, L, hidden_dim]
        
        # Cross-attention between text and video features
        text_proj = text_proj.unsqueeze(0).repeat(video_proj.size(0), 1, 1)  # [t, n, hidden_dim]
        text_enhanced = self.cross_attn(
            q=text_proj,  # [t, n, hidden_dim]
            k=video_proj,  # [t, L, hidden_dim]
            v=video_proj,  # [t, L, hidden_dim]
        )
        
        # Residual connection and output projection
        text_enhanced = self.output_norm(text_proj + text_enhanced)
        fused_cls_emb = self.output_proj(text_enhanced)  # [t, n, hidden_dim]
        
        # Generate segmentation prompts and class embeddings
        seg_emb = self.seg_proj(seg_emb).unsqueeze(0).expand(t, -1, -1)  # [t, 1, hidden_dim]
        fused_emb = torch.cat([seg_emb, fused_cls_emb], dim=1)  # [t, 1+n_cls, hidden_dim]
        
        seg_prompts, cls_emb = self.text_fc(fused_emb)
        seg_prompts = seg_prompts.permute(1, 0, 2)  # [1, t, sam_prompt_embed_dim]
        cls_emb = cls_emb[:, 1:, :]  # [t, n_cls, sam_prompt_embed_dim]
        return seg_prompts, cls_emb