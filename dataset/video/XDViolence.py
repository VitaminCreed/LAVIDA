# Code adapted from the anomalib library https://github.com/open-edge-platform/anomalib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
import torchvision
from torchvision.transforms.v2 import Transform
from torchvision.transforms.functional import resize
from torchvision.datasets.video_utils import _collate_fn, _VideoTimestampsDataset
from torchvision.datasets.utils import tqdm
from .base import ClipsIndexer, BaseVideoDataset
from pathlib import Path
from PIL import Image
import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, TypeVar, Union
from decord import VideoReader, cpu
import cv2

ANOMALY_CLASS = ['fighting', 'shooting', 'riot', 'abuse', 'car accident', 'explosion']

class XDViolenceDataset(BaseVideoDataset):
    def __init__(
        self,
        data_root: Path | str='',
        list_root: Path | str='./lists/XDViolence',
        **kwargs,
    )-> None:
        super().__init__(
            **kwargs,
        )
        
        self.data_root = Path(data_root)
        self.list_root = Path(list_root)
        self.indexer_cls = XDViolenceClipsIndexer
        self.samples = make_xdv_dataset(self.data_root, self.list_root, self.inference)
        

class XDViolenceClipsIndexer(ClipsIndexer):
    """
    The train and test subsets use different file formats, so separate clips
    indexer implementations are needed.
    """
    
    def _compute_frame_pts(self, metadata_path: Optional[str] = None) -> None:
        """Retrieve the number of frames in each video."""
        self.video_pts = []
        for video_path in self.video_paths:
            n_frame = len(VideoReader(video_path))
            self.video_pts.append(torch.tensor(range(n_frame), dtype=torch.long))

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
        
        vr = VideoReader(video_path, ctx=cpu(0))
        video = vr.get_batch(clip_pts).asnumpy() 
        video = torch.from_numpy(video).permute(0, 3, 1, 2)

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
            - 'anomaly_labels': Whether the current video clip is an anomaly. (Set as none in XD-Violence)
            - 'all_anomaly_types': List of all possible anomaly classes
            - 'gt_label': Tensor of frame-level anomaly labels (1=anomalous, 0=normal)
            - 'anomaly_masks': Pixel-level anomaly masks (1=anomalous, 0=normal)

            Returns None if video_shape is not provided when needed.
        """
        frames = self.clips[video_idx][frames_idx]
        anomaly_tracks = list(self.samples.anomaly_tracks)[video_idx]
        anomaly_mask_list = []
        
        gt_label = torch.zeros(len(frames), dtype=torch.long)
        for i, frame_idx in enumerate(frames):
            if anomaly_tracks is None:
                anomaly_mask = torch.zeros(video_shape, dtype=torch.uint8)
            else:
                anomaly_mask = torch.zeros(video_shape, dtype=torch.uint8)
                for track in anomaly_tracks:
                    start_frame, end_frame = track
                    if start_frame <= frame_idx <= end_frame:
                        anomaly_mask = torch.ones(video_shape, dtype=torch.uint8)
                        gt_label[i] = 1
            anomaly_mask_list.append(anomaly_mask)
        
        anomaly_masks = torch.stack(anomaly_mask_list, dim=0)  # [T, H, W]
        
        return {
            'anomaly_labels': None,
            'all_anomaly_types': ANOMALY_CLASS,
            'gt_label': gt_label,
            'anomaly_masks': anomaly_masks[::self.sample_interval], 
        }



def read_video_file(video_dir, label_map):
    video_dir = Path(video_dir)
    
    def extract_label(label_code: str) -> str:
        codes = [c for c in label_code.split('-') if c != '0']
        return codes[0] if codes else 'A' 
    
    video_data = []
    for mp4_file in video_dir.glob('*.mp4'):
        stem = mp4_file.stem 
        label_code = stem.split('_label_')[-1]
        
        video_data.append({
            'video_name': stem,
            'image_path': str(mp4_file.resolve()),
            'video_label': label_map.get(extract_label(label_code), 'unknown')
        })
    
    return pd.DataFrame(video_data)


def make_xdv_dataset(data_root: Path, list_root: Path, inference: bool):
    """Constructs a DataFrame for the XD-Violence anomaly detection dataset.

    Args:
        data_root: Path to the root directory containing XD-Violence dataset folders
        list_root: Path to directory containing dataset split information
        inference: Flag controlling which data to include:
                  - True: Process only test set with anomaly annotations
                  - False: Process only training set
                  - None: Process both training and test sets

    Returns:
        A pandas DataFrame containing:
        - video_name: Name of the video file
        - image_path: Path to the video file
        - video_label: Human-readable anomaly category (e.g., 'fighting', 'normal')
        - anomaly_tracks: List of [start,end] frame pairs for anomalous segments (None for normal videos and training set)
        - inference: Boolean flag indicating test set (True) or training set (False)
        - mask_path: Placeholder for mask paths (None in XD-Violence)

    """
    label_map = dict({
        'A': 'normal', 
        'B1': 'fighting', 
        'B2': 'shooting', 
        'B4': 'riot', 
        'B5': 'abuse', 
        'B6': 'car accident', 
        'G': 'explosion',
    })
    data_root = Path(data_root)
    
    if inference is True or inference is None:
        test_root = data_root / 'XD-Violence-test'
        annotaion_file = test_root / 'annotations.txt'
        test_video_root = test_root / 'videos'
        
        annotations = []
        with open(annotaion_file, 'r') as f:
            for line in f:
                if line.strip(): 
                    parts = line.strip().split()
                    video_name = parts[0]
                    time_pairs = np.array(parts[1:], dtype=int).reshape(-1, 2)
                    annotations.append({'video_name': video_name, 'anomaly_tracks': time_pairs})
        
        anno_df = pd.DataFrame(annotations)
        test_df = read_video_file(test_video_root, label_map)
        test_df = pd.merge(
            left=test_df,
            right=anno_df[['video_name', 'anomaly_tracks']], 
            on='video_name',
            how='left'
        )
        
        test_df['anomaly_tracks'] = test_df['anomaly_tracks'].where(
            cond=test_df['anomaly_tracks'].notna(),
            other=None
        )
        
        test_df['inference'] = True
        
    if inference is False or inference is None:
        train_root = data_root / 'XD-Violence-train'
        train_video_root = train_root / 'videos'
        train_df = read_video_file(train_video_root, label_map)
        train_df['inference'] = False
        train_df['anomaly_tracks'] = None
    
    if inference is True:
        result_df = test_df
    elif inference is False:
        result_df = train_df
    else:
        result_df = pd.concat([train_df, test_df])

    result_df['mask_path'] = None
        
    return result_df