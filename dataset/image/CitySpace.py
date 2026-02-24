import json
import random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
import torchvision
from torchvision.transforms.v2 import Transform
from .base import  BaseImageDataset
from pathlib import Path
from PIL import Image
from collections import defaultdict
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, TypeVar, Union
from ..data_utils import select_anomalies

category_dict = {19:'traffic light', 20: 'traffic sign', 22:'terrain', 
                 27:'truck' , 28:'bus', 31:'train', 32:'motorcycle', 33:'bicycle'}
ALL_CATEGORIES = list(category_dict.keys())


class CitySpaceDataset(BaseImageDataset):
    def __init__(
        self,
        inference: bool=False,
        data_root: Path | str='',
        # augmentations: Transform | None = None,
        anomaly_ratio: float=0.5,
        n_anomalies: int=1,
        **kwargs
    )-> None:
        super().__init__(
            # augmentations=augmentations,
            **kwargs
        )
        
        self.data_root = Path(data_root)
        self.anomaly_ratio = anomaly_ratio
        self.n_anomalies = n_anomalies
        self.samples = make_cityspace_dataset(self.data_root, inference)
        self.image_paths = self.samples.image_path
        self.mask_paths = self.samples.mask_path
        
    def get_image(self, idx: int) -> torch.Tensor:
        image_path = self.image_paths[idx]
        img = Image.open(image_path)
        img = torch.from_numpy(np.array(img)).to(torch.uint8)
        img = img.permute(2, 0, 1)
        return img
    
    def get_mask(self, idx: int, shape: torch.Tensor | None = None) -> torch.Tensor:
        mask_path = self.mask_paths[idx]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        mask = torch.from_numpy(mask).to(torch.uint8)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        is_anomaly = self.anomaly_ratio > random.random() and (torch.any(mask != 0))
        
        clip_object_class = torch.unique(mask).tolist()
        abnormal_classes = set(clip_object_class) & set(ALL_CATEGORIES)
        other_classes = list(set(ALL_CATEGORIES) - set(abnormal_classes))
        if len(abnormal_classes) == 0:
            is_anomaly = False
        elif len(other_classes) == 0:
            is_anomaly = True
            
        anomaly_labels, anomaly_types, abnormal_class = select_anomalies(
            anomaly_classes=abnormal_classes,
            all_classes=ALL_CATEGORIES,
            max_n_anomalies=self.n_anomalies,
            is_anomaly=is_anomaly,
            one_true_anomaly=True,
        )

        if is_anomaly:
            anomaly_masks = torch.where(mask == abnormal_class[0], torch.tensor(1), torch.tensor(0))
        else:
            anomaly_masks = torch.zeros_like(mask)
            
        anomaly_object_class = [category_dict[i] for i in anomaly_types]
        return {'anomaly_labels': anomaly_labels, 'all_anomaly_types': anomaly_object_class, 'anomaly_masks':anomaly_masks}


def get_path_pairs(img_folder: Path, mask_folder: Path, inference: bool = False):
    data = []
    for subdir in img_folder.iterdir():
        if not subdir.is_dir():  
            continue
        foldername = subdir.name

        for imgpath in subdir.glob('*.png'):
            if imgpath.suffix == '.png':
                """
                Example:
                    imgpath = Path("./Cityscapes/leftImg8bit/train/aachen/aachen_xxx_leftImg8bit.png")
                    foldername = "aachen"
                    maskname = "aachen_xxx_gtFine_labelIds.png"
                    maskpath = Path("./Cityscapes/gtFine/train/aachen/aachen_xxx_gtFine_labelIds.png")
                """
                maskname = imgpath.name.replace('leftImg8bit', 'gtFine_labelIds')
                maskpath = mask_folder / foldername / maskname

                if imgpath.is_file() and maskpath.is_file():
                    data.append({
                        'image_path': str(imgpath),  
                        'mask_path': str(maskpath), 
                        'inference': inference,
                    })
                else:
                    print('cannot find the mask or image:', imgpath, maskpath)

            
    data = pd.DataFrame(data)
    return data
    
def make_cityspace_dataset(root: Path | str,  inference: bool | None = None) -> pd.DataFrame:
    root = Path(root)
    train_img_folder = root / 'leftImg8bit' / 'train'
    train_mask_folder = root / 'gtFine' / 'train'
    val_img_folder = root /  'leftImg8bit' / 'val'
    val_mask_folder = root / 'gtFine'/ 'val'
    
    train_data = get_path_pairs(train_img_folder, train_mask_folder, inference=False)
    val_data = get_path_pairs(val_img_folder, val_mask_folder,  inference=True)
    
    if inference is not None:
        data = val_data if inference else train_data
    else:
        data = pd.concat([train_data, val_data])
    return data
    