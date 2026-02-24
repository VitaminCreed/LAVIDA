# Code adapted from the anomalib library https://github.com/open-edge-platform/anomalib
import json
import random
import torch
import torchvision
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, TypeVar, Union
from .base import ClipsIndexer, BaseVideoDataset
from ..data_utils import select_anomalies


class RefDavisDataset(BaseVideoDataset):
    def __init__(
        self,
        data_root: Path | str='',
        anno_file_path: Path | str='',
        list_root: Path | str='./lists/ReDavis',
        anomaly_ratio: float=0.5,
        n_anomalies: int=None,
        **kwargs,
    )-> None:
        super().__init__(**kwargs)
        
        self.data_root = Path(data_root)
        self.list_root = Path(list_root)
        self.anno_file_path = Path(anno_file_path)
        self.indexer_cls = RefDavisClipsIndexer
        self.anomaly_ratio = anomaly_ratio
        self.n_anomalies = n_anomalies
        self.samples = make_refdavis_dataset(self.data_root, self.anno_file_path, self.inference)
        
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
        self.indexer = RefDavisClipsIndexer( 
            video_paths=list(self.samples.image_path),
            mask_paths=list(self.samples.mask_path),
            samples=self.samples,
            clip_length_in_frames=self.clip_length_in_frames,
            frames_between_clips=self.frames_between_clips,
            _precomputed_metadata=meta_data,
            anomaly_ratio=self.anomaly_ratio,
            n_anomalies=self.n_anomalies,
        )
        

class RefDavisClipsIndexer(ClipsIndexer):
    """Clips indexer for RefDavis dataset.

    The train and test subsets use different file formats, so separate clips
    indexer implementations are needed.
    """
    def __init__(self, anomaly_ratio: float = 0.0, n_anomalies: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.anomaly_ratio = anomaly_ratio
        self.n_anomalies = n_anomalies
    
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
        video_idx, frames_idx = self.get_clip_location(idx)
        mask_file = self.mask_paths[video_idx] 
        
        if mask_file == "":  # no gt masks available for this clip
            return None
        frames = self.clips[video_idx][frames_idx]
        video_object_class = self.samples.expression[video_idx]  
        obj_id = self.samples.object_id[video_idx]

        mask_files = sorted(Path(mask_file).glob("*.png"))

        masks = []
        for pt in frames.int():
            mask_f = mask_files[pt]
            mask = np.array(Image.open(mask_f).convert('P'))
            mask = (mask==obj_id).astype(np.float32)
            masks.append(torch.from_numpy(mask))
        masks = torch.stack(masks, dim=0).squeeze(1)

        is_anomaly = self.anomaly_ratio > random.random() and (torch.any(masks != 0))
        if is_anomaly and torch.all(masks == 0):
            is_anomaly = False 
            
        selected_classes = self.samples.sample(self.n_anomalies)['expression'].tolist()
        
        anomaly_labels, anomaly_types = select_anomalies(
            anomaly_classes=[video_object_class],
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


def make_refdavis_dataset(
    root: Path,
    ann_file: Path,
    inference: Optional[bool] = None
) -> pd.DataFrame:
    """Generate RefDavis dataset metadata as a DataFrame.

    Args:
        root: Path to the dataset root directory
        ann_file: Path to the annotation JSON file
        inference: If True, only include inference set samples;
                  If False, only include training set samples;
                  If None, include all samples (default: None)

    Returns:
        pandas.DataFrame containing:
        - video: Video ID
        - expression: Referring expression
        - object_id: Target object ID
        - category: Object category
        - inference: Whether this is an inference sample (True/False)
        - image_path: Path to video frames (root/JPEGImages/480p/video_id)
        - mask_path: Path to semantic masks (root/Annotations_semantics/480p/video_id)
        - all_frames: List of all frame filenames (sorted)
    """
    # Read video metadata
    with open(root / 'meta.json', 'r') as meta_file:
        video_metas = json.load(meta_file)['videos']

    # Read expression annotations
    with open(ann_file, 'r') as annotation_file:
        video_expressions = json.load(annotation_file)['videos']
    
    video_ids = list(video_expressions.keys())
    metadata_list = []

    for video_id in video_ids:
        video_meta = video_metas[video_id]
        video_data = video_expressions[video_id]
        sorted_frames = sorted(video_data['frames'])
        image_path = root / 'JPEGImages' /  video_id
        mask_path = root / 'Annotations' / video_id

        for expression_id, expr in video_data['expressions'].items():
            is_inference = expr.get('is_test', False) 
            
            if inference is not None:
                if inference and not is_inference:
                    continue
                if not inference and is_inference:
                    continue

            metadata = {
                'video': video_id,
                'expression': expr['exp'],
                'object_id': int(expr['obj_id']),
                'category': video_meta['objects'][expr['obj_id']]['category'],
                'inference': is_inference,
                'image_path': str(image_path),
                'mask_path': str(mask_path),
                'all_frames': sorted_frames
            }
            metadata_list.append(metadata)

    df = pd.DataFrame(metadata_list)
    return df