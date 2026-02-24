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



ALL_CATEGORIES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road",
    "bed", "windowpane", "grass", "cabinet", "sidewalk",
    "person", "earth", "door", "table", "mountain", "plant",
    "curtain", "chair", "car", "water", "painting", "sofa",
    "shelf", "house", "sea", "mirror", "rug", "field", "armchair",
    "seat", "fence", "desk", "rock", "wardrobe", "lamp",
    "bathtub", "railing", "cushion", "base", "box", "column",
    "signboard", "chest of drawers", "counter", "sand", "sink",
    "skyscraper", "fireplace", "refrigerator", "grandstand",
    "path", "stairs", "runway", "case", "pool table", "pillow",
    "screen door", "stairway", "river", "bridge", "bookcase",
    "blind", "coffee table", "toilet", "flower", "book", "hill",
    "bench", "countertop", "stove", "palm", "kitchen island",
    "computer", "swivel chair", "boat", "bar", "arcade machine",
    "hovel", "bus", "towel", "light", "truck", "tower",
    "chandelier", "awning", "streetlight", "booth",
    "television receiver", "airplane", "dirt track", "apparel",
    "pole", "land", "bannister", "escalator", "ottoman", "bottle",
    "buffet", "poster", "stage", "van", "ship", "fountain",
    "conveyer belt", "canopy", "washer", "plaything",
    "swimming pool", "stool", "barrel", "basket", "waterfall",
    "tent", "bag", "minibike", "cradle", "oven", "ball", "food",
    "step", "tank", "trade name", "microwave", "pot", "animal",
    "bicycle", "lake", "dishwasher", "screen", "blanket",
    "sculpture", "hood", "sconce", "vase", "traffic light",
    "tray", "ashcan", "fan", "pier", "crt screen", "plate",
    "monitor", "bulletin board", "shower", "radiator", "glass",
    "clock", "flag"
]

BACKGROUND_CATEGORIES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road",
    "grass", "sidewalk", "earth", "mountain", "field", "sea", 
    "sand", "path", "runway", "river", "hill", "land", 
    "lake"
]

OBJ_CATEGORIES = [c for c in ALL_CATEGORIES if c not in BACKGROUND_CATEGORIES]


class ADE20KDataset(BaseImageDataset):
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
        self.samples = make_ade20k_dataset(self.data_root, inference)
        
    def __len__(self):
        return len(self.samples)
    
    
    def get_image(self, idx):
        image_path = self.samples.image_path[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image).permute(2, 0, 1)
    
    def get_mask(self, idx, shape = None):
        mask_path = self.samples.mask_path[idx]
        mask = np.array(Image.open(mask_path))
        mask[mask == 0] = 255
        mask -= 1
        mask[mask == 254] = 255
        
        unique_label = np.unique(mask).tolist()
        if 255 in unique_label:
            unique_label.remove(255)
            
        is_anomaly = (self.anomaly_ratio > random.random()) and (np.any(mask != 0))
        
        classes = [ALL_CATEGORIES[class_id] for class_id in unique_label]
        classes = [class_name for class_name in classes if class_name not in BACKGROUND_CATEGORIES]
        
        anomaly_labels, all_anomaly_classes, sampled_classes = select_anomalies(
            anomaly_classes=classes,
            all_classes=OBJ_CATEGORIES,
            max_n_anomalies=self.n_anomalies,
            is_anomaly=is_anomaly,
            one_true_anomaly=True,
        )
        assert self.n_anomalies + 1 > len(sampled_classes)
        if len(sampled_classes) == 0:
            return None
        
        masks = []
        for sampled_class in sampled_classes:
            class_id = ALL_CATEGORIES.index(sampled_class)
            mask = (mask == class_id).astype(np.uint8)
            masks.append(torch.from_numpy(mask))
        masks = torch.cat(masks, dim=0)
        
        if is_anomaly:
            anomaly_masks = masks
        else:
            anomaly_masks = torch.zeros_like(masks)

        return {'anomaly_labels': anomaly_labels, 'all_anomaly_types': all_anomaly_classes, 'anomaly_masks':anomaly_masks}


def make_ade20k_dataset(
    root: Path, 
    inference: bool | None = None
) -> pd.DataFrame:
    """Constructs a DataFrame for ADE20K semantic segmentation dataset.

    Args:
        root: Path to dataset root directory containing images/ and annotations/
        inference: If True, uses validation set; if False, uses training set
        
    Returns:
        DataFrame with columns:
        - image_path: Path to input image
        - mask_path: Path to segmentation mask
        - inference: Flag for train/val mode
    """
    root = Path(root)
    if inference:
        mode = 'validation'
    else:
        mode = 'training'
        
    images_dir = root / "images" / mode
    image_paths = sorted(images_dir.glob("*.jpg")) 
    ade20k_image_ids = [x.stem for x in image_paths]
    
    ade20k_images = [images_dir / f"{image_id}.jpg" for image_id in ade20k_image_ids]
    ade20k_labels = [
        (root / "annotations" / mode / f"{image_id}.png") 
        for image_id in ade20k_image_ids
    ]
    
    df = pd.DataFrame({
        'image_path': [str(img) for img in ade20k_images],
        'mask_path': [str(mask) for mask in ade20k_labels],
        'inference': [inference] * len(ade20k_images)
    })
    
    print(f"ade20k: {len(ade20k_images)} images")
    return df

    