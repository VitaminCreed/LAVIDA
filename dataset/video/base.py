# Code adapted from the anomalib library https://github.com/open-edge-platform/anomalib
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import cv2
import torch
from torchvision.datasets.video_utils import VideoClips
from torch.utils.data import Dataset
from torchvision.tv_tensors import Mask
from torchvision.transforms.v2 import Transform, functional
from torchvision.transforms import InterpolationMode
import pandas as pd
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, TypeVar, Union
import json
import numpy as np
import decord
import random
from transformers import AutoProcessor
from ..messages_template import create_structured_template
from qwen_vl_utils.vision_process import smart_resize, VIDEO_MIN_PIXELS, VIDEO_TOTAL_PIXELS, VIDEO_MAX_PIXELS, FRAME_FACTOR, IMAGE_FACTOR


def qwen_resize_video(
    video, 
    resized_height=None, 
    resized_width=None, 
    min_pixels=None, 
    max_pixels=None, 
    total_pixels=None,
    image_factor=IMAGE_FACTOR,
):
    nframes, _, height, width = video.shape
    min_pixels = min_pixels if min_pixels else VIDEO_MIN_PIXELS
    total_pixels = total_pixels if total_pixels else VIDEO_TOTAL_PIXELS
    cal_max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
    max_pixels = max_pixels if max_pixels else cal_max_pixels
    if resized_height and resized_width:
        resized_height, resized_width = smart_resize(
            resized_height,
            resized_width,
            factor=image_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    video = functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    return video


class ClipsIndexer(VideoClips, ABC):
    """Extension of torchvision's VideoClips class for video and mask indexing.

    This class extends ``VideoClips`` to handle both video frames and their 
    corresponding mask annotations. 

    Subclasses must implement the ``get_mask`` method. The default implementation
    assumes ``video_paths`` contains video files. For custom data formats
    (e.g., image sequences), subclasses should override ``get_clip`` and
    ``_compute_frame_pts``.

    Args:
        video_paths: List of paths to video files in the dataset.
        mask_paths: List of paths to mask files corresponding to each video.
        samples: Complete dataset metadata dataframe.
        clip_length_in_frames: Temporal length of output clips in original frames.
            Calculated as: total_sampled_frames * sample_interval.
        frames_between_clips: Stride between clip start frames. Defaults to clip_length_in_frames (no overlap).
        sample_interval: Frame sampling interval.
        _precomputed_metadata: Pre-loaded video metadata for initialization. Structure should match torchvision. VideoClips class requirements.
    """

    def __init__(
        self,
        video_paths: list[str],
        mask_paths: list[str],
        samples: pd.DataFrame,
        clip_length_in_frames: int = 1,
        frames_between_clips: int = 1,
        sample_interval: int = 1,
        _precomputed_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            video_paths=video_paths,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            _precomputed_metadata=_precomputed_metadata,
            output_format="TCHW",
        )
        self.mask_paths = mask_paths
        self.samples = samples
        self.sample_interval = sample_interval

    def last_frame_idx(self, video_idx: int) -> int:
        """Get index of the last frame in a video.

        Args:
            video_idx: Index of the video in the dataset

        Returns:
            Index of the last frame
        """
        return self.clips[video_idx][-1][-1].item()

    @abstractmethod
    def get_mask(self, idx: int) -> torch.Tensor | None:
        """Get masks for the clip at the given index.

        Args:
            idx: Index of the clip

        Returns:
            Dict containing mask and labels
        """
        raise NotImplementedError
    
        
    def get_item(self, idx: int) -> dict:
        """Get video clip and metadata at the given index.

        Args:
            idx: Index of the clip to retrieve

        Returns:
            VideoItem containing the clip frames, masks, path and metadata
        """
        with warnings.catch_warnings():
            # silence warning caused by bug in torchvision
            # see https://github.com/pytorch/vision/issues/5787
            warnings.simplefilter("ignore")
            clip_result = self.get_clip(idx)
            if clip_result is None:
                return None
            clip, _, _, _ = clip_result

        video_idx, clip_idx = self.get_clip_location(idx)
        video_path = self.video_paths[video_idx]
        clip_pts = (self.clips[video_idx][clip_idx][::self.sample_interval] / self.sample_interval).int()
        video_shape = clip.shape[-2:]

        gt = self.get_mask(idx, video_idx, clip_idx, video_shape)
        if gt is None:
            return None
        gt_mask = {'anomaly_masks': gt['anomaly_masks']}
        all_anomaly_types = gt['all_anomaly_types']
        anomaly_labels = gt['anomaly_labels']
        gt_label = gt['gt_label']
        
        
        return {
            "images": clip,
            "video_path": video_path,
            "frame_idxs": clip_pts - min(clip_pts),  
            "gt_mask": gt_mask,
            "gt_label": gt_label,
            "all_anomaly_types": all_anomaly_types,
            "anomaly_labels": anomaly_labels,
        }
    
    def _save_metadata(self, metadata_path: str) -> None:
        """
        Save video metadata (video_paths, video_pts, video_fps) to a JSON file.

        Args:
            metadata_path (str): Path to save the metadata.
        """
        metadata = {
            "video_paths": self.video_paths,
            "video_pts": [pts.tolist() for pts in self.video_pts],  
            "video_fps": self.video_fps,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
    
    def _init_from_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Initialize the object from saved metadata.

        Args:
            metadata (Dict[str, Any]): Dictionary containing video_paths, video_pts, and video_fps.
        """
        self.video_paths = metadata["image_path"]
        assert len(self.video_paths) == len(metadata["video_pts"])
        self.video_pts = [torch.tensor(pts, dtype=torch.long) for pts in metadata["video_pts"]]
        assert len(self.video_paths) == len(metadata["video_fps"])
        self.video_fps = metadata["video_fps"]

        
        
class BaseVideoDataset(Dataset, ABC):
    """Base class for video dataset processing.

    Args:
        total_sampled_frames (int): Total number of frames to sample per video clip.
        sample_interval (int): Interval between sampled frames (e.g., 1=every frame, 2=every other frame).
        llm_sample_frames (int): Number of video frames to use as input for LLM processing. 
        frames_between_clips (int): Interval frames between consecutive clips when splitting long videos.
        augmentations (Transform, optional): Data augmentation transforms to apply to video clips. 
            Defaults to None (no augmentation).
    """

    def __init__(
        self,
        inference: bool,
        total_sampled_frames: int,
        llm_sample_frames: int,
        frames_between_clips: int,
        sample_interval: int = 1,
        augmentations: Transform | None = None,
    ) -> None:
        super().__init__()

        self.clip_length_in_frames = total_sampled_frames * sample_interval
        self.frames_between_clips = frames_between_clips * sample_interval
        self.llm_sample_frames = llm_sample_frames
        self.sample_interval = sample_interval
        self.augmentations = augmentations
        self.total_sampled_frames = total_sampled_frames
        self.inference = inference

        self.indexer: ClipsIndexer | None = None
        self.indexer_cls: Callable | None = None
        self._samples: pd.DataFrame | None = None

    def __len__(self) -> int:
        """Get length of the dataset.

        Returns:
            int: Number of clips in the dataset.

        Raises:
            TypeError: If ``self.indexer`` is not an instance of ``ClipsIndexer``.
        """
        if not isinstance(self.indexer, ClipsIndexer):
            msg = "self.indexer must be an instance of ClipsIndexer."
            raise TypeError(msg)
        return self.indexer.num_clips()
    
    @property
    def samples(self) -> pd.DataFrame:
        """Get the samples DataFrame.

        Returns:
            DataFrame: DataFrame containing dataset samples.

        Raises:
            RuntimeError: If samples DataFrame has not been set.
        """
        if self._samples is None:
            msg = (
                "Dataset does not have a samples dataframe. Ensure that a dataframe has "
                "been assigned to `dataset.samples`."
            )
            raise RuntimeError(msg)
        return self._samples

    @samples.setter
    def samples(self, samples: pd.DataFrame) -> None:
        """Overwrite samples and re-index subvideos.

        Args:
            samples (DataFrame): DataFrame with new samples.

        Raises:
            ValueError: If the indexer class is not set.
        """
        self._samples = samples
        self._setup_clips()
    
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
        
        self.indexer = self.indexer_cls(  # pylint: disable=not-callable
            video_paths=list(self.samples.image_path),
            mask_paths=list(self.samples.mask_path),
            samples=self.samples,
            clip_length_in_frames=self.clip_length_in_frames,
            frames_between_clips=self.frames_between_clips,
            sample_interval=self.sample_interval, 
            _precomputed_metadata=meta_data,
        )
    
    def __getitem__(self, index: int) -> dict:
        """Get the dataset item for the index.

        Args:
            index (int): Index of the item to be returned.

        Returns:
            Dict: A dictionary containing the dataset item.
                keys:
                    'images': Inputs for SAM2, shape [T, 3, H, W].
                    'video_path': Path to the video file. 
                    'frame_idxs': Dict, contains "cond_frame_idxs" and "non_cond_frame_idxs". 
                    'all_anomaly_types': List of anomaly types for the video. 
                    'pixel_values': Image inputs for LLM, it is None in video dataset.
                    'pixel_values_videos': Video inputs for LLM. 
                    'gt_label': Frame level ground truth label for the video.
                    'gt_mask': Pixel level ground truth mask for the video. 
                    'message': Prompts for LLM.

        Raises:
            TypeError: If ``self.indexer`` is not an instance of ``ClipsIndexer``.
        """
        if not isinstance(self.indexer, ClipsIndexer):
            msg = "self.indexer must be an instance of ClipsIndexer."
            raise TypeError(msg)
        item = self.indexer.get_item(index)
        if item is None:
            return self.__getitem__(0)

        # extract frames for LLM
        all_frame_idx = item['frame_idxs'].tolist()
        cond_idx = np.linspace(0, len(all_frame_idx)-1, self.llm_sample_frames, dtype=int)
        cond = [all_frame_idx[i] for i in cond_idx]
        non_cond = [f for f in all_frame_idx if f not in cond]
        item['frame_idxs'] = {
            'all_frame_idx': all_frame_idx,
            'cond_frame_idx': cond,         # LLM sample frames and conditional frames for SAM2
            'non_cond_frame_idx': non_cond   # non-conditional frames for SAM2
        }
        if self.inference:
            item["pixel_values_videos"] = qwen_resize_video(
                video=item["images"][item['frame_idxs']['cond_frame_idx']],
                min_pixels=160*28*28*self.llm_sample_frames, # 256*28*28, 
                max_pixels=192*28*28*self.llm_sample_frames, # 512*28*28,  
            )
        else:
            item["pixel_values_videos"] = qwen_resize_video(
                video=item["images"][item['frame_idxs']['cond_frame_idx']],
                min_pixels=32*28*28*self.llm_sample_frames, # 256*28*28, 
                max_pixels=64*28*28*self.llm_sample_frames, # 512*28*28,  
            )
        # apply transforms
        if item["gt_mask"] is not None:
            if self.augmentations:
                item["images"], item["gt_mask"]['anomaly_masks'] = self.augmentations(
                    item["images"], Mask(item["gt_mask"]['anomaly_masks']))
            if item["gt_label"] is None:
                item["gt_label"] = (item["gt_mask"]['anomaly_masks'] == 1).any(dim=(1, 2)).int()

        else:
            if self.augmentations:
                item["images"] = self.augmentations(item["images"])
            if item["gt_label"] is None:
                item["gt_label"] = torch.zeros(item["images"].shape[0]).int()
        
        # generate messages for LLM
        item["message"] = create_structured_template(
            path='', 
            type='video',
            nframes=self.llm_sample_frames,
            anomaly_list=item["all_anomaly_types"],
        )

        item["pixel_values"] = None
        return item
 
    
    