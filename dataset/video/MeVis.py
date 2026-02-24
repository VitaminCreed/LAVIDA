# Code adapted from the anomalib library https://github.com/open-edge-platform/anomalib
import json
import random
import torch.nn.functional as F

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torchvision.transforms.v2 import Transform
from .base import ClipsIndexer, BaseVideoDataset
from pathlib import Path
from PIL import Image
from pycocotools import mask as coco_mask

from typing import Any, Callable, cast, Dict, List, Optional, Tuple, TypeVar, Union
from ..data_utils import select_anomalies


class MeVisDataset(BaseVideoDataset):
    def __init__(
        self,
        data_root: Path | str='',
        list_root: Path | str='./lists/UBnormal',
        anomaly_ratio: float=0.5,
        n_anomalies: int=None,
        **kwargs
    )-> None:
        super().__init__(**kwargs)
        
        self.data_root = Path(data_root)
        self.list_root = Path(list_root)
        self.indexer_cls = MeVisClipsIndexer
        self.anomaly_ratio = anomaly_ratio
        self.n_anomalies = n_anomalies
        self.samples = make_mevis_dataset(self.data_root, self.inference)
        
    def _setup_clips(self) -> None:
        """Compute the video and frame indices of the subvideos.

        Should be called after each change to ``self._samples``.

        Raises:
            TypeError: If ``self.indexer_cls`` is not callable.
        """
        if not callable(self.indexer_cls):
            msg = "self.indexer_cls must be callable."
            raise TypeError(msg)

        if 'video_pts' in self.samples.columns:
            meta_data =  self.samples[["image_path", "video_pts", "video_fps"]].to_dict(orient="list")
        else:
            meta_data = None
        
        # self.indexer = self.indexer_cls(  # pylint: disable=not-callable
        self.indexer = MeVisClipsIndexer( 
            video_paths=list(self.samples.image_path),
            mask_paths=list(self.samples.mask_path),
            samples=self.samples,
            clip_length_in_frames=self.clip_length_in_frames,
            frames_between_clips=self.frames_between_clips,
            _precomputed_metadata=meta_data,
            anomaly_ratio=self.anomaly_ratio,
            n_anomalies=self.n_anomalies,
        )



class MeVisClipsIndexer(ClipsIndexer):
    """Clips indexer for ShanghaiTech test dataset.

    The train and test subsets use different file formats, so separate clips
    indexer implementations are needed.
    """
    def __init__(self, anomaly_ratio: float = 0.0, n_anomalies: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.anomaly_ratio = anomaly_ratio
        self.n_anomalies = n_anomalies
        mask_path = self.samples.mask_path[0]
        with open(mask_path) as fp:
            self.mask_dict = json.load(fp)
    
    def _compute_frame_pts(self, metadata_path: Optional[str] = None) -> None:
        """Retrieve the number of frames in each video."""
        self.video_pts = []
        for video_path in self.video_paths:
            n_frames = len(list(Path(video_path).glob("*.jpg")))
            self.video_pts.append(torch.Tensor(range(n_frames)))

        self.video_fps = [1] * len(self.video_paths)
    
    def get_clip(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], int]:
        """Get a subclip from a list of videos.

        Args:
            idx (int): Index of the subclip. Must be between 0 and num_clips().

        Returns:
            tuple containing:
                - video (torch.Tensor): Video clip tensor
                - audio (torch.Tensor): Empty audio tensor
                - info (dict): Empty info dictionary
                - video_idx (int): Index of the video in video_paths

        Raises:
            IndexError: If idx is out of range.
        """
        if idx >= self.num_clips():
            msg = f"Index {idx} out of range ({self.num_clips()} number of clips)"
            raise IndexError(msg)
        video_idx, clip_idx = self.get_clip_location(idx)
        video_path = self.video_paths[video_idx]
        clip_pts = self.clips[video_idx][clip_idx] 
        clip_pts = clip_pts[::self.sample_interval]

        frames = sorted(Path(video_path).glob("*.jpg"))
        frame_paths = [frames[pt] for pt in clip_pts.int()]
        video = []
        for frame_path in frame_paths:
            frame = Image.open(frame_path).convert("RGB")
            frame_tensor = torch.from_numpy(np.array(frame)).permute(2, 0, 1)  # HWC -> CHW
            video.append(frame_tensor)
        video = torch.stack(video)

        return video, torch.empty((1, 0)), {}, video_idx

    def get_mask(self, 
                idx: int,
                video_idx: int,
                frames_idx: int,
                video_shape: torch.Size | None = None,
        ) -> dict[str, torch.Tensor] | None:
        """Generates anomaly masks and labels.
    
        Args:
            idx: Index of the clip in the dataset (unused in current implementation)
            video_idx: Index of the video in the samples collection
            frames_idx: Indices of frames within the video to process
            video_shape: Expected shape (H, W) of the output masks. If None, will raise error.
                        Defaults to None.

        Returns:
            A dictionary containing:
            - 'anomaly_labels': Whether the current video clip is an anomaly. 
            - 'all_anomaly_types': List of all possible anomaly classes
            - 'gt_label': Tensor of frame-level anomaly labels (1=anomalous, 0=normal)
            - 'anomaly_masks': Pixel-level anomaly masks (1=anomalous, 0=normal)

            Returns None if video_shape is not provided when needed.
        """
        
        video_idx, clip_idx = self.get_clip_location(idx)
        mask_file = self.mask_paths[video_idx]
        clip_pts = self.clips[video_idx][clip_idx].int().tolist()
        
        sample = self.samples.iloc[video_idx]
        all_exp = sample.exp
        if len(all_exp) == 1:
            exp_id = 0
        else:
            exp_id = random.randint(0, len(all_exp) - 1)
        expression = all_exp[exp_id]
        anno_ids = sample.anno_id[exp_id]
        masks = torch.zeros(len(clip_pts), *video_shape[-2:])
        for anno_id in anno_ids:
            for f_idx, f in enumerate(clip_pts):
                frame_anno = self.mask_dict[anno_id][f]
                if frame_anno is not None:
                    masks[f_idx] += torch.from_numpy(coco_mask.decode(frame_anno)) 

        is_anomaly = self.anomaly_ratio > random.random()
        if is_anomaly and torch.all(masks == 0):
            is_anomaly = False  # if this video does not have gt masks, set is_anomaly to False
        
        selected_classes = self.samples.sample(self.n_anomalies)['exp'].tolist()
        selected_classes = [random.choice(x) for x in selected_classes]
        anomaly_labels, anomaly_types = select_anomalies(
            anomaly_classes=[expression],
            all_classes=selected_classes,
            max_n_anomalies=self.n_anomalies,
            is_anomaly=is_anomaly,
            one_true_anomaly=False,
        )
    
        if is_anomaly:
            anomaly_masks = masks
        else:
            anomaly_masks = torch.zeros_like(masks)
            
        gt_label = (anomaly_masks == 1).any(dim=(1, 2)).int()
        return {
            'anomaly_labels': anomaly_labels, 
            'all_anomaly_types': anomaly_types, 
            'gt_label': gt_label,
            'anomaly_masks':anomaly_masks[::self.sample_interval], 
        }
    


def make_mevis_dataset(data_root: Path, inference: bool):
    if inference:
        data_root = data_root / "valid"
    else:
        data_root = data_root / "train"
    exp_meta_path = data_root / "meta_expressions.json"
    mask_dict_path = data_root / "mask_dict.json"
        
    with open(exp_meta_path, 'r') as f:
        subset_expressions_by_video = json.load(f)['videos']
        
    videos = list(subset_expressions_by_video.keys())
    metas = []
    for vid in videos:
        vid_data = subset_expressions_by_video[vid]
        vid_frames = sorted(vid_data['frames'])
        vid_len = len(vid_frames)
        meta = {}
        meta['video'] = vid
        meta['frames'] = vid_len
        meta['exp'] = []
        meta['image_path'] = data_root / "JPEGImages" / vid
        meta['mask_path'] = mask_dict_path
        meta['inference'] = inference

        cnt = 0
        meta['exp'] = []
        meta['obj_id'] = []
        meta['anno_id'] = []
        for exp_id, exp_dict in vid_data['expressions'].items():
            
            meta['exp'].append(exp_dict['exp'])
            meta['obj_id'].append([int(x) for x in exp_dict['obj_id']])
            meta['anno_id'].append([str(x) for x in exp_dict['anno_id']])
            cnt += 1
            if cnt == 8:
                break
        if len(meta['exp']) > 0:
            metas.append(meta)
    return pd.DataFrame(metas)