import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.tv_tensors import Mask
from torchvision.transforms.v2 import Transform, functional
from torchvision.transforms import InterpolationMode
import pandas as pd
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, TypeVar, Union
import json
import numpy as np
import random
from transformers import AutoProcessor
from ..messages_template import create_structured_template
from qwen_vl_utils.vision_process import smart_resize, MIN_PIXELS, MAX_PIXELS, IMAGE_FACTOR


def qwen_resize_image(
    image, 
    resized_height=None, 
    resized_width=None, 
    min_pixels=None, 
    max_pixels=None, 
    image_factor=IMAGE_FACTOR,
):
    _, _, height, width = image.shape
    min_pixels = min_pixels if min_pixels else MIN_PIXELS
    max_pixels = max_pixels if max_pixels else MAX_PIXELS
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
    image = functional.resize(
        image,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    return image


class BaseImageDataset(Dataset, ABC):
    def __init__(
        self,
        augmentations: Transform | None = None,
    ) -> None:
        self.augmentations = augmentations
        self.samples: pd.DataFrame | None = None
        
    def __len__(self) -> int:
        """Get length of the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        if not isinstance(self.samples, pd.DataFrame):
            msg = "Samples are not inited. "
            raise TypeError(msg)
        return len(self.samples)
    
    @abstractmethod
    def get_image(self, idx: int) -> torch.Tensor | None:
        """Get image at the given index.

        Args:
            idx: Index of the image

        Returns:
            Tensor containing image.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_mask(self, idx: int, shape: torch.Tensor | None = None) -> torch.Tensor | None:
        """Get masks for the image at the given index.

        Args:
            idx: Index of the image

        Returns:
            Dict containing mask, labels.
        """
        raise NotImplementedError
    
    
    def __getitem__(self, index: int) -> dict:
        """Get the dataset item for the index.

        Args:
            index (int): Index of the item to be returned.

        Returns:
            Dict: A dictionary containing the dataset item.
                keys:
                    'images': Inputs for SAM2, shape [1, 3, H, W].
                    'video_path': Path to the video file, it is None in image dataset.. 
                    'frame_idxs': Dict, contains "cond_frame_idxs" and "non_cond_frame_idxs", it is None in image dataset. 
                    'all_anomaly_types': List of anomaly types for the image. 
                    'pixel_values': Image inputs for LLM.
                    'pixel_values_videos': Video inputs for LLM, it is None in image dataset. 
                    'gt_label': Frame level ground truth label for the video.
                    'gt_mask': Pixel level ground truth mask for the video. 
                    'message': Prompts for LLM.
        """
        item = {}
        item["images"] = self.get_image(index).unsqueeze(0)
        image_shape = item["images"].shape
        gt = self.get_mask(index, image_shape)
        if gt is None:
            return self.__getitem__(0)
        item["gt_mask"] = {'anomaly_masks': gt['anomaly_masks'].unsqueeze(0)}
        item["all_anomaly_types"] = gt["all_anomaly_types"]
        item["anomaly_labels"] = gt["anomaly_labels"]
        
        item["pixel_values"] = qwen_resize_image(
            image=item["images"],
            min_pixels=32*28*28,  # 4
            max_pixels=1024*28*28,  # 16384
        )
        if item["gt_mask"] is not None:
            if self.augmentations:
                item["images"], item["gt_mask"]['anomaly_masks'] = self.augmentations(
                    item["images"], Mask(item["gt_mask"]['anomaly_masks']))
            item["gt_label"] = (item["gt_mask"]['anomaly_masks'] == 1).any(dim=(1, 2)).int()
        else:
            if self.augmentations:
                item["images"] = self.augmentations(item["images"])
            item["gt_label"] = torch.zeros(item["images"].shape[0]).int()
            
        item["message"] = create_structured_template(
            path='', 
            type='image',
            anomaly_list=item["all_anomaly_types"],
        )

        item["video_path"] = None
        # item["pixel_values"] = None
        # item['frame_idxs'] = {
        #     'all_frame_idx': [0],
        #     'cond_frame_idx': [0],         # LLM sample frames and conditional frames for SAM2
        #     'non_cond_frame_idx': []  # non-conditional frames for SAM2
        # }
        item["pixel_values_videos"] = None
        item['frame_idxs'] = None
        return item