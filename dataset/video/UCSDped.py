# Code adapted from the anomalib library https://github.com/open-edge-platform/anomalib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torchvision.transforms.v2 import Transform
from .base import ClipsIndexer, BaseVideoDataset
from pathlib import Path
from typing import Any
from ..data_utils import validate_path, read_mask, read_image
from torchvision.datasets.video_utils import _collate_fn, _VideoTimestampsDataset
from torchvision.datasets.utils import tqdm
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, TypeVar, Union
import json
import os
from PIL import Image
import tifffile


ANOMALY_CLASS = ["car", "cart", "wheelchair", "skate", "bicycle"]



class UCSDpedDataset(BaseVideoDataset):
    """UCSDped Dataset class.

    Args:
        split (Split): Dataset split - either ``Split.TRAIN`` or ``Split.TEST``
        root (Path | str): Path to the root directory containing the dataset.
            Defaults to ``"./datasets/ucsped"``.
        scene (int): Index of the dataset scene (category) in range [1, 13].
            Defaults to ``1``.
        clip_length_in_frames (int, optional): Number of frames in each video
            clip. Defaults to ``2``.
        frames_between_clips (int, optional): Number of frames between each
            consecutive video clip. Defaults to ``1``.
        target_frame (VideoTargetFrame): Specifies which frame in the clip to use
            for ground truth retrieval. Defaults to ``VideoTargetFrame.LAST``.
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.

    Example:
        >>> from anomalib.data.datasets import UCSDpedDataset
        >>> from anomalib.data.utils import Split
        >>> dataset = UCSDpedDataset(
        ...     root="./datasets/ucsped",
        ...     scene=1,
        ...     split=Split.TRAIN
        ... )
    """

    def __init__(
        self,
        data_root: Path | str='',
        list_root: Path | str='',
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.root = Path(data_root)
        self.list_root = Path(list_root)
        self.indexer_cls = UCSDpedClipsIndexer
        self.samples = make_ucsdped_dataset(self.root, self.inference)
        

class UCSDpedClipsIndexer(ClipsIndexer):
    """Clips indexer for UCSDped test dataset.

    The train and test subsets use different file formats, so separate clips
    indexer implementations are needed.
    """

    def get_mask(self, idx, video_idx, frames_idx, video_shape):
        """Retrieve masks from BMP files or return zero masks if no mask available."""
        frames = self.clips[video_idx][frames_idx]
        T = len(frames)  
        mask_path = self.mask_paths[video_idx]
        if mask_path is None:
            H, W = video_shape[-2], video_shape[-1]
            zero_mask = torch.zeros((T, H, W), dtype=torch.uint8)
            return {
                'anomaly_labels': None,
                'all_anomaly_types': ANOMALY_CLASS,
                'gt_label': torch.zeros(T, dtype=torch.int),
                'anomaly_masks': zero_mask,
            }
        
        mask_files = sorted(Path(mask_path).glob("*.bmp"))
        if not mask_files or len(mask_files) <= frames.max():
            return None  
        
        masks = []
        for frame_idx in frames:
            img = Image.open(mask_files[int(frame_idx)])#.convert('L')
            if video_shape:
                img = img.resize((video_shape[-1], video_shape[-2]), Image.NEAREST)
            img = (np.array(img) > 0).astype(np.uint8)
            masks.append(torch.from_numpy(img))
        
        anomaly_masks = torch.stack(masks)
        gt_target = (anomaly_masks == 1).any(dim=(1, 2)).int()
        
        anomaly_masks = anomaly_masks[::self.sample_interval]
        return {
            'anomaly_labels': None,
            'all_anomaly_types': ANOMALY_CLASS,
            'gt_label': gt_target,
            'anomaly_masks': anomaly_masks,
            'normal_masks': torch.zeros_like(anomaly_masks),
        }

    def _compute_frame_pts(self) -> None:
        """Retrieve the number of frames in each video."""
        self.video_pts = []
        for video_path in self.video_paths:
            n_frames = len(list(Path(video_path).glob("*.tif")))
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

        frames = sorted(Path(video_path).glob("*.tif"))

        if len(frames) == 0:
            raise FileNotFoundError(f"No TIF frames found in {video_path}")
        
        frame_paths = [frames[pt] for pt in clip_pts.int()]
        
        video_frames = []
        for frame_path in frame_paths:
            try:
                img = tifffile.imread(str(frame_path))
            except:
                return None

            img_array = np.array(img)
        
            if img_array.ndim == 2: 
                img_array = np.expand_dims(img_array, axis=-1) 
            elif img_array.ndim == 3 and img_array.shape[2] > 3:  
                img_array = img_array[:, :, :3]
            
            frame_tensor = torch.from_numpy(img_array).permute(2, 0, 1).repeat(3, 1, 1)
            video_frames.append(frame_tensor)
        
        video = torch.stack(video_frames)
        return video, torch.empty((1, 0)), {}, video_idx


def make_ucsdped_dataset(root: Path,  inference: bool | None = None) -> DataFrame:
    """Generates a DataFrame containing video directory information for the UCSD Pedestrian dataset.
    
    Processes either training data, testing data, or both based on the inference flag,
    and constructs a DataFrame with paths to video frames and corresponding ground truth masks.

    Args:
        root: Path to the root directory containing UCSDped1 and UCSDped2 subdirectories
        inference: Controls which subset to return:
                  - True: Only returns test set data
                  - False: Only returns training set data
                  - None: Returns both training and test data (default)
    
    Returns:
        A pandas DataFrame with the following columns:
        - image_path: Path to directory containing video frames (str)
        - inference: Boolean flag indicating test set (True) or training set (False)
        - mask_path: Path to ground truth masks directory (None for training set)
        
    """
    rows = []
    
    # Process each dataset directory (UCSDped1, UCSDped2)
    for dataset_dir in root.iterdir():
        if not dataset_dir.is_dir():
            continue
            
        # Process training set (when inference is False or None)
        if inference is None or inference is False:
            train_dir = dataset_dir / 'Train'
            if train_dir.exists() and train_dir.is_dir():
                for video_dir in train_dir.iterdir():
                    if video_dir.is_dir() and video_dir.name.startswith('Train'):
                        rows.append({
                            'image_path': str(video_dir),
                            'inference': False,
                            'mask_path': None  # Training set has no ground truth
                        })
        
        # Process test set (when inference is True or None)
        if inference is None or inference is True:
            test_dir = dataset_dir / 'Test'
            if test_dir.exists() and test_dir.is_dir():
                for video_dir in test_dir.iterdir():
                    # Only process video directories (exclude ground truth folders)
                    if video_dir.is_dir() and video_dir.name.startswith('Test') and '_gt' not in video_dir.name:
                        # Find corresponding ground truth directory
                        gt_dir = test_dir / f"{video_dir.name}_gt"
                        mask_path = str(gt_dir) if gt_dir.exists() and gt_dir.is_dir() else None
                        
                        rows.append({
                            'image_path': str(video_dir),
                            'inference': True,
                            'mask_path': mask_path
                        })
    
    return pd.DataFrame(rows)