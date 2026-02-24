# Code adapted from the anomalib library https://github.com/open-edge-platform/anomalib
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
import torchvision
from torchvision.transforms.v2 import Transform
from .base import ClipsIndexer, BaseVideoDataset
from pathlib import Path
from PIL import Image
from itertools import chain
from collections import defaultdict
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, TypeVar, Union

from ..data_utils import select_anomalies


category_dict = {
    'airplane': 0, 'ape': 1, 'bear': 2, 'bike': 3, 'bird': 4, 'boat': 5, 'bucket': 6, 'bus': 7, 'camel': 8, 'cat': 9,
    'cow': 10, 'crocodile': 11, 'deer': 12, 'dog': 13, 'dolphin': 14, 'duck': 15, 'eagle': 16, 'earless_seal': 17,
    'elephant': 18, 'fish': 19, 'fox': 20, 'frisbee': 21, 'frog': 22, 'giant_panda': 23, 'giraffe': 24, 'hand': 25,
    'hat': 26, 'hedgehog': 27, 'horse': 28, 'knife': 29, 'leopard': 30, 'lion': 31, 'lizard': 32, 'monkey': 33,
    'motorbike': 34, 'mouse': 35, 'others': 36, 'owl': 37, 'paddle': 38, 'parachute': 39, 'parrot': 40, 'penguin': 41,
    'person': 42, 'plant': 43, 'rabbit': 44, 'raccoon': 45, 'sedan': 46, 'shark': 47, 'sheep': 48, 'sign': 49,
    'skateboard': 50, 'snail': 51, 'snake': 52, 'snowboard': 53, 'squirrel': 54, 'surfboard': 55, 'tennis_racket': 56,
    'tiger': 57, 'toilet': 58, 'train': 59, 'truck': 60, 'turtle': 61, 'umbrella': 62, 'whale': 63, 'zebra': 64, 
}
ALL_CATEGORIES = list(category_dict.keys())


class RefVosDataset(BaseVideoDataset):
    def __init__(
        self,
        data_root: Path | str='',
        exp_meta_path: Path | str='',
        list_root: Path | str='./lists/RefVos',
        anomaly_ratio: float=0.5,
        n_anomalies: int=None,
        **kwargs,
    )-> None:
        super().__init__(**kwargs)
        
        self.data_root = Path(data_root)
        self.list_root = Path(list_root)
        self.exp_meta_path = Path(exp_meta_path)
        self.indexer_cls = RefVosClipsIndexer
        self.anomaly_ratio = anomaly_ratio
        self.n_anomalies = n_anomalies
        self.samples = make_refvos_dataset(self.data_root, self.exp_meta_path, self.inference)
        
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
        self.indexer = RefVosClipsIndexer( 
            video_paths=list(self.samples.image_path),
            mask_paths=list(self.samples.mask_path),
            samples=self.samples,
            clip_length_in_frames=self.clip_length_in_frames,
            frames_between_clips=self.frames_between_clips,
            _precomputed_metadata=meta_data,
            anomaly_ratio=self.anomaly_ratio,
            n_anomalies=self.n_anomalies,
        )



class RefVosClipsIndexer(ClipsIndexer):
    """Clips indexer for RefVos dataset.

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
        
        video_idx, frames_idx = self.get_clip_location(idx)
        mask_file = self.mask_paths[video_idx] 
        
        mask_files = sorted(Path(mask_file).glob("*.png"))
        frame_name_id = [f.stem[:5] for f in mask_files]
        
        if mask_file == []:  # no gt masks available for this clip
            return None
        frames = self.clips[video_idx][frames_idx]
        frames_object_class = self.samples.frame_obj[video_idx]
        exp = self.samples.exp[video_idx]

        masks = []
        objs = []
        for pt in frames.int():
            f_id = frame_name_id[pt]
            if f_id in frames_object_class.keys(): 
                objs.extend(frames_object_class[f_id])

            mask_f = mask_files[pt]
            mask = Image.open(mask_f).convert('P')
            mask = np.array(mask).astype(np.float32)
            if mask.shape[0] == 3:
                assert mask[0] == mask[1] == mask[2]
                mask = mask[:1, :, :]
            masks.append(torch.from_numpy(mask))
        
        objs = list(set(objs))
        if objs:
            unique_obj_ids = random.choices(objs, k=random.randint(1, 2))
        else:
            unique_obj_ids = []
        obj_exp = [random.choice(exp[id]) for id in unique_obj_ids]

        if len(masks) == 0:
            return None
        masks = torch.stack(masks, dim=0).squeeze(1)
        masks = torch.isin(masks, torch.tensor(unique_obj_ids)).float()
        
        is_anomaly = self.anomaly_ratio > random.random() and (torch.any(masks != 0))
        sampled_rows = self.samples.sample(n=self.n_anomalies)['exp']
        sampled_classes = []
        for d in sampled_rows:
            all_strings = [s for lst in d.values() for s in lst]
            sampled_classes.append(random.choice(all_strings))

        anomaly_labels, anomaly_types = select_anomalies(
            anomaly_classes=obj_exp,
            all_classes=sampled_classes,
            max_n_anomalies=self.n_anomalies,
            is_anomaly=is_anomaly,
            one_true_anomaly=False,
        )
        
        if is_anomaly:
            anomaly_masks = masks
            normal_masks = torch.zeros_like(masks)
        else:
            anomaly_masks = torch.zeros_like(masks)
            normal_masks = masks
        gt_label = (anomaly_masks == 1).any(dim=(1, 2)).int()
        return {
            'anomaly_labels': anomaly_labels, 
            'all_anomaly_types': anomaly_types, 
            'gt_label': gt_label,
            'anomaly_masks':anomaly_masks[::self.sample_interval], 
        }
    

def make_refvos_dataset(
    root: Path, 
    exp_meta_path: Path, 
    inference: bool = False
) -> DataFrame:
    """Constructs a DataFrame dataset for RefVOS.

    Args:
        root: Path to the root directory 
        exp_meta_path: Path to JSON file containing referring expressions metadata
        inference: If True, uses 'valid' subset, otherwise uses 'train' subset.
                  Defaults to False (training mode).

    Returns:
        A pandas DataFrame where each row represents a video with:
        - image_path: Directory path containing video frames (str)
        - mask_path: Directory path containing segmentation masks (str)
        - frames: Total number of frames in the video (int)
        - exp: Dictionary mapping object IDs to referring expressions (dict)
        - frame_obj: Dictionary mapping frames to contained object IDs (dict)
        - inference: Flag indicating train/validation mode (bool)
    """

    def reorganize_by_frames(objects):
        """Reorganizes object-centric data into frame-centric structure.
        
        Converts {obj_id: {category: [frame_numbers]}} to {frame_number: [obj_ids]} format.

        Args:
            objects: Dictionary of objects with their appearing frames

        Returns:
            Dictionary mapping frame numbers to sorted lists of object IDs
        """
        frames_dict = defaultdict(set)  # Using set for automatic deduplication
        for obj_id, obj_data in objects.items():
            obj_id_int = int(obj_id)
            for frame in obj_data["frames"]:
                frames_dict[frame].add(obj_id_int)
        
        # Convert to {frame: [obj_id, ...]} with sorted IDs
        return {frame: sorted(obj_ids) for frame, obj_ids in frames_dict.items()}
    

    if inference == False:
        root = Path(root) / 'train'
    else:
        root = Path(root) / 'valid'
    
    meta_path = Path(root) / 'meta.json'
    

    with open(meta_path, 'r') as f:
        subset_metas_by_video = json.load(f)['videos']

    with open(exp_meta_path, 'r') as f:
        subset_expressions_by_video = json.load(f)['videos']
    videos = list(subset_expressions_by_video.keys())
    
    all_metas = []
    for vid in videos:
        vid_meta = subset_metas_by_video[vid]
        vid_exp_data = subset_expressions_by_video[vid]
        vid_frames = sorted(vid_exp_data['frames'])  
        vid_len = len(vid_frames)
        
        video_dir = root / 'JPEGImages' / vid 
        mask_dir = root / 'Annotations' / vid

        exp = defaultdict(list)
        for exp_dict in vid_exp_data['expressions'].values():
            exp[int(exp_dict['obj_id'])].append(exp_dict['exp'])

        frame_obj = reorganize_by_frames(vid_meta['objects'])
                
        meta = {
            'image_path': str(video_dir),   # Path to video frames directory
            'mask_path':  str(mask_dir),    # Path to segmentation masks directory
            'frames': vid_len,              # Total frame count
            'exp': exp,                    # Object ID to expressions mapping
            'frame_obj': frame_obj,         # Frame to object IDs mapping
            'inference': inference,         # Train/validation mode flag
        }
        all_metas.append(meta)
        
    return pd.DataFrame(all_metas)