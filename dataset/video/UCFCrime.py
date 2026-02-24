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

ANOMALY_CLASS = ['abuse', 'arrest', 'arson', 'assault', 'burglary', 'explosion', 'fighting',  
                 'road accidents', 'robbery', 'shooting', 'shoplifting', 'stealing', 'vandalism']



class UCFCrimeDataset(BaseVideoDataset):
    def __init__(
        self,
        data_root: Path | str='',
        list_root: Path | str='./lists/UCFCrime',
        **kwargs,
    )-> None:
        super().__init__(
            **kwargs,
        )
        
        self.data_root = Path(data_root)
        self.list_root = Path(list_root)
        self.indexer_cls = UCFCrimeClipsIndexer
        self.samples = make_ucfcime_dataset(self.data_root, self.list_root, self.inference)
        

class UCFCrimeClipsIndexer(ClipsIndexer):
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
        # metadata_path='/root/autodl-tmp/program/0224_ub/lists/UBnormal/UBnormal_test_metadata.json'
        # if metadata_path:
        #     self._save_metadata(metadata_path)
    
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

    def get_mask(
        self, 
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
            - 'anomaly_labels': Whether the current video clip is an anomaly. (Set as none in UCFCrime)
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
            'gt_label': gt_label,
            'all_anomaly_types': ANOMALY_CLASS,
            'anomaly_masks': anomaly_masks[::self.sample_interval], 
        }

        
def make_ucfcime_dataset(data_root: Path, list_root: Path, inference: bool):
    """
    Create a dataset DataFrame for UCF-Crime videos with anomaly annotations.
    
    Processes either training data, testing data, or both based on the inference flag,
    and constructs a DataFrame containing video metadata and anomaly information.
    
    Args:
        data_root: Path to the root directory containing video folders
        list_root: Path to directory containing train/test split files (unused in current implementation)
        inference: Boolean flag to select data type:
                  - True: process only testing data
                  - False: process only training data
                  - None: process both training and testing data
    
    Returns:
        pandas.DataFrame containing:
        - video_name: Name of the video file
        - image_path: Full path to the video file
        - mask_path: Placeholder for mask path (None in current implementation)
        - video_label: Category label for the video
        - anomaly_tracks: List of anomaly time intervals (for anomalous videos) or None
        - inference: Boolean flag indicating if this is test data
    """
    data_root = Path(data_root)
    anomaly_root = data_root / "Anomaly-Videos"
    normal_train_root = data_root / "Training_Normal_Videos_Anomaly"
    normal_test_root = data_root / "Testing_Normal_Videos_Anomaly"
    
    train_videos_file = data_root / "Anomaly_Train.txt"
    test_videos_file = data_root / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
    
    
    if inference is True or inference is None:
        test_df = pd.read_csv(test_videos_file, sep='\s+', header=None, 
                            names=['video_name', 'video_label', 'start1', 'end1', 'start2', 'end2'])
        
        test_df['anomaly_tracks'] = test_df.apply(
            lambda row: np.array([[row['start1'], row['end1']], [row['start2'], row['end2']]]), 
            axis=1
        )
        
        test_df['image_path'] = test_df.apply(
            lambda row: (
                str(anomaly_root/row['video_label']/row['video_name'])
                if row['video_label'].lower() != "normal" 
                else str(normal_test_root/row['video_name'])
            ),
            axis=1
        )
        
        test_df['inference'] = True
        test_df['mask_path'] = None
        test_df = test_df[['video_name', 'image_path', 'mask_path', 'video_label', 'anomaly_tracks', 'inference']]
    
    if inference is False or inference is None:
        train_file_list = pd.read_csv(train_videos_file, header=None, names=['video_name'])
        train_df = pd.DataFrame()
        
        for _, row in train_file_list.iterrows():
            video_name = row['video_name']
            folder_name = os.path.dirname(video_name)  
            file_name = os.path.basename(video_name)  
            
            if folder_name != "Training_Normal_Videos_Anomaly":
                new_row = {
                    'video_name': video_name,
                    'image_path': str(anomaly_root / video_name),
                    'video_label': folder_name,
                    'mask_path': None,
                    'anomaly_tracks': None,
                    'inference': False
                }
            else:
                new_row = {
                    'video_name': video_name,
                    'image_path': str(normal_train_root / file_name), 
                    'video_label': "Normal",
                    'mask_path': None,
                    'anomaly_tracks': None,
                    'inference': False
                }
            
            train_df = pd.concat([train_df, pd.DataFrame([new_row])], ignore_index=True)
        
    if inference is True:
        result_df = test_df
    elif inference is False:
        result_df = train_df
    else:
        result_df = pd.concat([train_df, test_df], ignore_index=True)
        
    return result_df