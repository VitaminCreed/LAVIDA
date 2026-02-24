# Code adapted from the anomalib library https://github.com/open-edge-platform/anomalib
from pathlib import Path
import pandas as pd
import torch
from pandas import DataFrame
from .base import ClipsIndexer, BaseVideoDataset
from pathlib import Path
import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, TypeVar, Union
from decord import VideoReader, cpu
import cv2


ANOMALY_CLASS = ['running', 'jumping','falling', 'fighting', 'sleeping', 'crawling', 'having a seizure', 
                 'laying down', 'dancing', 'stealing', 'rotating 360 degrees', 'shuffling', 'injured', 
                 'drunk', 'stumbling walk', 'car accident', 'fire', 'smoke', 'jaywalking', 'driving outside lane']


class UBnormalDataset(BaseVideoDataset):
    def __init__(
        self,
        data_root: Path | str='',
        list_root: Path | str='./lists/UBnormal',
        only_anomaly_video: bool=True,
        **kwargs,
    )-> None:
        super().__init__(**kwargs)
        
        self.data_root = Path(data_root)
        self.list_root = Path(list_root)
        self.indexer_cls = UBnormalClipsIndexer
        self.samples = make_ubnormal_dataset(self.data_root, self.list_root, self.inference, only_anomaly_video)
        
        
class UBnormalClipsIndexer(ClipsIndexer):
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
            - 'anomaly_labels': Whether the current video clip is an anomaly. 
            - 'all_anomaly_types': List of all possible anomaly classes
            - 'gt_label': Tensor of frame-level anomaly labels (1=anomalous, 0=normal)
            - 'anomaly_masks': Pixel-level anomaly masks (1=anomalous, 0=normal)

            Returns None if video_shape is not provided when needed.
        """
        mask_folder = self.mask_paths[video_idx]
        frames = self.clips[video_idx][frames_idx]
        anomaly_tracks = list(self.samples.anomaly_tracks)[video_idx]
        video_name = os.path.basename(mask_folder).replace("_annotations", "")
        anomaly_mask_list = []
        all_mask_list = []

        gt_label = torch.zeros(len(frames), dtype=torch.long)
        for i, frame_idx in enumerate(frames):
            mask_file_name = f"{video_name}_{frame_idx:04d}_gt.png"
            mask_file_path = Path(mask_folder) / mask_file_name

            mask = cv2.imread(str(mask_file_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask file not found: {mask_file_path}")
            
            mask = torch.from_numpy(mask).to(torch.uint8)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            
            if video_shape is not None and mask.shape != video_shape:
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),  
                    size=(video_shape[0], video_shape[1]),
                    mode='nearest'
                ).squeeze(0).squeeze(0).to(torch.uint8)
        
            all_mask = (mask > 0).to(torch.uint8)
            
            if anomaly_tracks is None:
                anomaly_mask = torch.zeros_like(mask, dtype=torch.uint8)
            else:
                anomaly_mask = torch.zeros_like(mask, dtype=torch.uint8)
                for track in anomaly_tracks:
                    obj_id, start_frame, end_frame = track
                    if start_frame <= frame_idx <= end_frame:
                        anomaly_mask[mask == obj_id] = 1
                        gt_label[i] = 1
            
            all_mask_list.append(all_mask)
            anomaly_mask_list.append(anomaly_mask)
        
        all_masks = torch.stack(all_mask_list, dim=0)  # [T, H, W]
        anomaly_masks = torch.stack(anomaly_mask_list, dim=0)  # [T, H, W]

        return {
            'anomaly_labels': 1 if torch.any(anomaly_masks == 1) else 0,
            'all_anomaly_types': ANOMALY_CLASS,
            'gt_label': gt_label,
            'anomaly_masks': anomaly_masks[::self.sample_interval], 
        }

        
def make_ubnormal_dataset(data_root: Path, list_root: Path, inference: bool, only_anomaly_video: bool = False):
    """
    Create a DataFrame containing video metadata for the UBnormal dataset.
    
    Args:
        data_root: Path to the directory containing video data
        list_root: Path to the directory containing train/test split files
        inference: Whether this is for inference (test) or training
        only_anomaly_video: If True, only include abnormal videos in the output
        
    Returns:
        A DataFrame containing:
        - video_name: Name of the video
        - video_path: Path to the video file
        - annotation_path: Path to annotation files
        - inference: Flag indicating if this is for inference
        - video_label: Binary label (1 for abnormal, 0 for normal)
        - anomaly_track: List of anomaly tracks if available
    """
    if inference:
        normal_file = list_root / 'normal_test_video_names.txt'
        abnormal_file = list_root / 'abnormal_test_video_names.txt'
        meta_file = list_root / 'UBnormal_test_metadata.json'
    else:
        raise NotImplementedError
        
    abnormal_df = generate_video_df(abnormal_file, data_root, inference)
    
    if not only_anomaly_video:
        normal_df = generate_video_df(normal_file, data_root, inference)
        df = pd.concat([abnormal_df, normal_df], ignore_index=True)
    else:
        df = abnormal_df
        
    if os.path.exists(meta_file):
        metadata = pd.read_json(meta_file)
        df = pd.merge(df, metadata, on="video_name", how="inner")
    return df


def generate_video_df(txt_file_path: Path, root_folder: Path, inference: bool = False):
    """
    Generate a DataFrame from a text file containing video names.
    
    Args:
        txt_file_path: Path to text file containing video names (one per line)
        root_folder: Root directory where video files are stored
        inference: Flag to indicate if this is for inference
        
    Returns:
        A DataFrame containing:
        - video_name: Name of the video
        - image_path: Path to the video file (as string)
        - mask_path: Path to annotation directory (as string)
        - inference: Pass-through inference flag
        - video_label: 1 for abnormal videos, 0 for normal
        - anomaly_tracks: List of anomaly tracks for abnormal videos
    """
    with txt_file_path.open('r') as file:
        video_names = file.read().splitlines()

    data = []

    for video_name in video_names:
        scene_folder = root_folder / f"Scene{video_name.split('_')[2]}"
        video_path = scene_folder / f"{video_name}.mp4"
        annotation_path = scene_folder / f"{video_name}_annotations"

        is_abnormal = video_name.startswith('abnormal')
        video_label = 1 if is_abnormal else 0

        anomaly_track = []
        if is_abnormal:
            tracks_path = annotation_path / f"{video_name}_tracks.txt"
            if tracks_path.exists():
                with tracks_path.open('r') as f:
                    anomaly_track = [list(map(lambda x: int(float(x)), line.strip().split(','))) 
                                    for line in f if line.strip()]

        data.append({
            'video_name': video_name,
            'image_path': str(video_path),
            'mask_path': str(annotation_path),
            'inference': inference,
            'video_label': video_label,
            'anomaly_tracks': anomaly_track
        })
    
    return pd.DataFrame(data)