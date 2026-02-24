import os
import random

import cv2
import numpy as np
import torch
from pycocotools import mask
from pathlib import Path
import pandas as pd

from torchvision.transforms.v2 import Transform
from pathlib import Path

from .base import  BaseImageDataset
from .refer import REFER
from .grefer import G_REFER
from ..data_utils import select_anomalies


class CocoDataset(BaseImageDataset):
    def __init__(
        self, 
        inference: bool=False,
        data_root: Path | str='',
        data_list: list = [],
        anomaly_ratio: float=0.5,
        n_anomalies: int=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data_root = Path(data_root)
        self.anomaly_ratio = anomaly_ratio
        self.n_anomalies = n_anomalies
        self.refer_seg_data = make_coco_dataset(self.data_root, data_list, inference)
        self.lengths = [ds['len'] for ds_name, ds in self.refer_seg_data.items()]
        
    def __len__(self):
        return sum(self.lengths)
    
    def get_idx(self, idx):
        for i, ds_name in enumerate(self.refer_seg_data.keys()):
            if idx < self.lengths[i]:
                return ds_name, idx
            idx -= self.lengths[i]
        raise IndexError("Index out of range")
    
    def sample_other_anomalies(self, idx, num_anomalies):
        ds_name, idx = self.get_idx(idx)
        all_refs = self.refer_seg_data[ds_name]["img2refs"]
        all_indices = list(self.refer_seg_data[ds_name]["img2refs"].keys())
        valid_indices = [i for i in all_indices if i != idx] 
        if len(valid_indices) > num_anomalies:
            sampled_indices = random.sample(valid_indices, num_anomalies)
        else:
            sampled_indices = valid_indices
        sampled_refs = [all_refs[i] for i in sampled_indices]
        
        sents = []
        for refs in sampled_refs:
            ref = random.choice(refs)
            sent = random.choice(ref["sentences"])
            sents.append(sent["sent"])
        sents = random.sample(sents, k=num_anomalies)
        return sents
    
    def get_image(self, idx):
        ds_name, idx = self.get_idx(idx)
        image_path = self.refer_seg_data[ds_name]['images'][idx]["file_name"]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image).permute(2, 0, 1)
    
    def get_mask(self, idx, shape = None):
        ds_name, idx = self.get_idx(idx)
        image_info = self.refer_seg_data[ds_name]['images'][idx]
        image_id = self.refer_seg_data[ds_name]['images'][idx]["id"]
        annotations = self.refer_seg_data[ds_name]['annotations']
        refs = self.refer_seg_data[ds_name]["img2refs"][image_id]
        
        select_ref = random.choice(refs)
        sents = []
        ann_ids = []
        
        ann_ids.append(select_ref["ann_id"])
        sent = random.choice(select_ref["sentences"])
        sents.append(sent["sent"])
        
        sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
        sampled_classes = sampled_sents
        
        flag = False
        masks = []
        for ann_id in sampled_ann_ids:
            if isinstance(ann_id, list):
                flag = True
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                else:
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = annotations[ann_id_i]

                        if len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        else:
                            if type(ann["segmentation"][0]) == list:
                                rle = mask.frPyObjects(
                                    ann["segmentation"],
                                    image_info["height"],
                                    image_info["width"],
                                )
                            else:
                                rle = ann["segmentation"]
                                for i in range(len(rle)):
                                    if not isinstance(rle[i]["counts"], bytes):
                                        rle[i]["counts"] = rle[i]["counts"].encode()
                            m = mask.decode(rle)
                            m = np.sum(
                                m, axis=2
                            )  
                            m = m.astype(np.uint8)  
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue

            ann = annotations[ann_id]

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:
                rle = mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = mask.decode(rle)
            m = np.sum(
                m, axis=2
            ) 
            m = m.astype(np.uint8)
            masks.append(m)

        masks = torch.from_numpy(np.concatenate(masks, axis=0))
        
        is_anomaly = (self.anomaly_ratio > random.random()) and (torch.any(masks != 0))
        assert self.n_anomalies + 1 > len(sampled_classes)
        anomaly_types = self.sample_other_anomalies(idx, self.n_anomalies)
        
        anomaly_labels, anomaly_types = select_anomalies(
            anomaly_classes=sampled_classes,
            all_classes=anomaly_types,
            max_n_anomalies=self.n_anomalies,
            is_anomaly=is_anomaly,
            one_true_anomaly=False,
        )
        
        
        if len(sampled_classes) > 1:
            id_class = random.choice(range(len(sampled_classes)))
            sampled_classes = [sampled_classes[id_class]]
            masks = masks[id_class : id_class + 1]
            
        if is_anomaly:
            anomaly_masks = masks
        else:
            anomaly_masks = torch.zeros_like(masks)


        return {'anomaly_labels': anomaly_labels, 'all_anomaly_types': anomaly_types, 'anomaly_masks':anomaly_masks}
    

def make_coco_dataset(
    root: Path | str, 
    refer_seg_ds_list: list, 
    inference: bool | None = None
) -> pd.DataFrame:
    """
    Args:
        root (Path | str): root directory of the dataset
        refer_seg_ds_list (list): list of refer segmentation datasets ["refclef", "refcocog", "refcocog+", "refcocog++"]
        inference (bool | None): whether to use inference mode
    """
    split = 'val' if inference else 'train'

    DATA_DIR = Path(root)
    refer_seg_data = {}
    for ds in refer_seg_ds_list:
        if ds == "refcocog":
            splitBy = "umd"
        else:
            splitBy = "unc"

        if ds == "grefcoco":
            refer_api = G_REFER(DATA_DIR, ds, splitBy)
        else:
            refer_api = REFER(DATA_DIR, ds, splitBy)
        ref_ids_train = refer_api.getRefIds(split=split)
        images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
        refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)

        refer_seg_ds = {}
        refer_seg_ds["images"] = []
        loaded_images = refer_api.loadImgs(image_ids=images_ids_train)

        for item in loaded_images:
            item = item.copy()
            if ds == "refclef":
                item["file_name"] = os.path.join(
                    DATA_DIR, "images/saiapr_tc-12", item["file_name"]
                )
            else:
                item["file_name"] = os.path.join(
                    DATA_DIR, "images/mscoco/images/train2014", item["file_name"]
                )
            refer_seg_ds["images"].append(item)
        refer_seg_ds["annotations"] = refer_api.Anns  # anns_train
        refer_seg_ds["len"] = len(refer_seg_ds["images"])

        img2refs = {}
        for ref in refs_train:
            image_id = ref["image_id"]
            img2refs[image_id] = img2refs.get(image_id, []) + [
                ref,
            ]
        refer_seg_ds["img2refs"] = img2refs
        refer_seg_data[ds] = refer_seg_ds
    return refer_seg_data




