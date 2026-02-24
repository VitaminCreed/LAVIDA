import os
import random

import cv2
import numpy as np
import torch
from pycocotools import mask
from pathlib import Path
import pandas as pd
from PIL import Image
import pickle as pkl

from torchvision.transforms.v2 import Transform
from pathlib import Path

from .base import  BaseImageDataset
from ..data_utils import select_anomalies



class CocoStuffDataset(BaseImageDataset):
    def __init__(
        self, 
        inference: bool=False,
        data_root: Path | str='',
        anomaly_ratio: float=0.5,
        n_anomalies: int=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data_root = Path(data_root)
        self.anomaly_ratio = anomaly_ratio
        self.n_anomalies = n_anomalies
        self.samples = make_cocostuff_dataset(self.data_root, inference)
        class_file = Path("./dataset/image/cocostuff_classes.txt")
        cocostuff_classes = []
        with class_file.open() as f:
            for line in f.readlines()[1:]:
                cocostuff_classes.append(line.strip().split(": ")[-1])
        self.cocostuff_classes = cocostuff_classes
        self.cocostuff_class2index = {
            c: i for i, c in enumerate(self.cocostuff_classes)
        }
    
        
    def __len__(self):
        return len(self.samples)
    
    def get_image(self, idx):
        image_path = self.samples.image_path[idx]
        import pdb; pdb.set_trace()
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image).permute(2, 0, 1)
    
    def get_mask(self, idx, shape = None):
        mask_path = self.samples.mask_path[idx]
        mask = np.array(Image.open(mask_path))
        for c, i in self.cocostuff_class2index.items():
            if "-" in c:
                mask[mask == i] = 255
        
        unique_label = np.unique(mask).tolist()
        if 255 in unique_label:
            unique_label.remove(255)
            
        is_anomaly = (self.anomaly_ratio > random.random()) and (np.any(mask != 0))
        
        classes = [self.cocostuff_classes[class_id] for class_id in unique_label]
        
        anomaly_labels, all_anomaly_classes, sampled_classes = select_anomalies(
            anomaly_classes=classes,
            all_classes=self.cocostuff_classes,
            max_n_anomalies=self.n_anomalies,
            is_anomaly=is_anomaly,
            one_true_anomaly=True,
        )
        assert self.n_anomalies + 1 > len(sampled_classes)
        if len(sampled_classes) == 0:
            return None
        
        masks = []
        for sampled_class in sampled_classes:
            class_id = self.cocostuff_classes.index(sampled_class)
            mask = (mask == class_id).astype(np.uint8)
            masks.append(torch.from_numpy(mask))
        masks = torch.cat(masks, dim=0)
        
        if is_anomaly:
            anomaly_masks = masks
            normal_masks = torch.zeros_like(masks)
        else:
            anomaly_masks = torch.zeros_like(masks)
            normal_masks = masks

        return {'anomaly_labels': anomaly_labels, 'all_anomaly_types': all_anomaly_classes, 'anomaly_masks':anomaly_masks, 'normal_masks':normal_masks}


def make_cocostuff_dataset(
    root: Path, 
    inference: bool | None = None
) -> pd.DataFrame:
    base_path = Path(root)
    
    if inference:
        image_dir = base_path / "images" / "val2017"
        mask_dir = base_path / "annotations" / "stuffthingmaps_trainval2017" / "val2017"
    else:
        image_dir = base_path / "images" / "train2017"
        mask_dir = base_path / "annotations" / "stuffthingmaps_trainval2017" / "train2017"
    image_paths = list(image_dir.glob("*.jpg"))
    mask_paths = list(mask_dir.glob("*.png"))
    image_paths = [
        str(image_dir / path.with_suffix(".jpg").name)
        for path in mask_paths
    ]
    df = pd.DataFrame({
        "image_path": image_paths,
        "mask_path": [str(p) for p in mask_paths],
        "inference": [inference] * len(mask_paths)
    })
    return  df

    