from pathlib import Path
import numpy as np
import random
from typing import List, Union, Optional, Dict
from torchvision.transforms.v2 import Transform
from torch.utils.data import Dataset

from .video import (
    UBnormalDataset,
    ShanghaiTechDataset,
    VosDataset,
    RefVosDataset,
    MeVisDataset,
    UCFCrimeDataset,
    XDViolenceDataset,
    RefDavisDataset,
    UCSDpedDataset,
)

from .image import (
    CitySpaceDataset,
    CocoDataset,
    ADE20KDataset,
    MapillaryDataset,
    CocoStuffDataset
)


class HybridDataset(Dataset):
    def __init__(
        self,
        inference: bool = False,
        datasets: List[str] = [],
        sampling_ratios: Optional[Dict[str, float]] = None,
        avenue_args: Dict = {},
        ubnormal_args: Dict = {},
        shanghaitech_args: Dict = {},
        ucfcrime_args: Dict = {},
        xdviolence_args: Dict = {},
        ucsdped_args: Dict = {},
        vos_args: Dict = {},
        refvos_args: Dict = {},
        mevis_args: Dict = {},
        refdavis_args: Dict = {},
        cityspace_args: Dict = {},
        coco_args: Dict = {},
        ade20k_args: Dict = {},
        mapillary_args: Dict = {},
        list_root: Union[Path, str] = "./lists",
        total_sampled_frames: int = 8,
        frames_between_clips: int = 8,
        llm_sample_frames: int = 8,
        augmentations: Union[Transform, None] = None,
        anomaly_ratio: float = 1.,
    ) -> None:
        super().__init__()

        self.datasets = []
        # Initialize individual datasets
        for dataset in datasets:
            dataset = dataset.lower()   # all in lowercase
            if dataset == "ubnormal":
                self.datasets.append(UBnormalDataset(
                    inference=inference,
                    list_root=Path(list_root) / 'UBnormal',
                    total_sampled_frames=total_sampled_frames,
                    frames_between_clips=frames_between_clips,
                    llm_sample_frames=llm_sample_frames,
                    augmentations=augmentations,
                    **ubnormal_args,
                ))
            elif dataset == "shanghaitech":
                self.datasets.append(ShanghaiTechDataset(
                    inference=inference,
                    list_root=Path(list_root) / 'ShanghaiTech',
                    total_sampled_frames=total_sampled_frames,
                    frames_between_clips=frames_between_clips,
                    llm_sample_frames=llm_sample_frames,
                    augmentations=augmentations,
                    **shanghaitech_args,
                ))
            elif dataset == "ucfcrime":
                self.datasets.append(UCFCrimeDataset(
                    inference=inference,
                    list_root=Path(list_root) / 'UCFCrime',
                    total_sampled_frames=total_sampled_frames,
                    frames_between_clips=frames_between_clips,
                    llm_sample_frames=llm_sample_frames,
                    augmentations=augmentations,
                    **ucfcrime_args,
                ))
            elif dataset == "xdviolence":
                self.datasets.append(XDViolenceDataset(
                    inference=inference,
                    list_root=Path(list_root) / 'XDViolence',
                    total_sampled_frames=total_sampled_frames,
                    frames_between_clips=frames_between_clips,
                    llm_sample_frames=llm_sample_frames,
                    augmentations=augmentations,
                    **xdviolence_args,
                ))
            elif dataset == "ucsdped":
                self.datasets.append(UCSDpedDataset(
                    inference=inference,
                    list_root=Path(list_root) / 'UCSDped',
                    total_sampled_frames=total_sampled_frames,
                    frames_between_clips=frames_between_clips,
                    llm_sample_frames=llm_sample_frames,
                    augmentations=augmentations,
                    **ucsdped_args,
                ))
            elif dataset == "vos":
                self.datasets.append(VosDataset(
                    inference=inference,
                    list_root=Path(list_root) / 'Vos',
                    total_sampled_frames=total_sampled_frames,
                    frames_between_clips=frames_between_clips,
                    llm_sample_frames=llm_sample_frames,
                    augmentations=augmentations,
                    anomaly_ratio= anomaly_ratio,
                    **vos_args,
                ))
            elif dataset == "refvos":
                self.datasets.append(RefVosDataset(
                    inference=inference,
                    list_root=Path(list_root) / 'RefVos',
                    total_sampled_frames=total_sampled_frames,
                    frames_between_clips=frames_between_clips,
                    llm_sample_frames=llm_sample_frames,
                    augmentations=augmentations,
                    anomaly_ratio= anomaly_ratio,
                    **refvos_args,
                ))
            elif dataset == 'mevis':
                self.datasets.append(MeVisDataset(
                    inference=inference,
                    list_root=Path(list_root) / 'MeVis',
                    total_sampled_frames=total_sampled_frames,
                    frames_between_clips=frames_between_clips,
                    llm_sample_frames=llm_sample_frames,
                    augmentations=augmentations,
                    anomaly_ratio= anomaly_ratio,
                    **mevis_args,
                ))
            elif dataset == "refdavis":
                self.datasets.append(RefDavisDataset(
                    inference=inference,
                    list_root='',
                    total_sampled_frames=total_sampled_frames,
                    frames_between_clips=frames_between_clips,
                    llm_sample_frames=llm_sample_frames,
                    augmentations=augmentations,
                    anomaly_ratio= anomaly_ratio,
                    **refdavis_args,
                ))
            elif dataset == 'cityspace':
                self.datasets.append(CitySpaceDataset(
                    inference=inference,
                    augmentations=augmentations,
                    anomaly_ratio= anomaly_ratio,
                    **cityspace_args,
                ))
            elif dataset == 'coco':
                self.datasets.append(CocoDataset(
                    inference=inference,
                    augmentations=augmentations,
                    anomaly_ratio= anomaly_ratio,
                    **coco_args,
                ))
            elif dataset == 'ade20k':
                self.datasets.append(ADE20KDataset(
                    inference=inference,
                    augmentations=augmentations,
                    anomaly_ratio= anomaly_ratio,
                    **ade20k_args,
                ))
            elif dataset == 'mapillary':
                self.datasets.append(MapillaryDataset(
                    inference=inference,
                    augmentations=augmentations,
                    anomaly_ratio= anomaly_ratio,
                    **mapillary_args,
                ))
            else:
                raise ValueError(f'Unknown dataset {dataset}')
        
        self.lengths = [len(d) for d in self.datasets]
        self.dataset_num = len(self.lengths)
        
        self.sampling_ratios = sampling_ratios
        if self.sampling_ratios is not None:
            assert set(self.sampling_ratios.keys()) == set(datasets)
            ratio_list = [self.sampling_ratios[ds] for ds in datasets]
            
            total_ratio = sum(ratio_list)
            assert abs(total_ratio - 1.) < 1e-6
            self.normalized_ratios = [r / total_ratio for r in ratio_list]
            
            self.cumulative_ratios = np.cumsum(self.normalized_ratios)
        else:
            self.cumulative_ratios = None
        

    def __len__(self) -> int:
        return sum(self.lengths)

    def __getitem__(self, idx: int):
        if self.cumulative_ratios is None:
            for i in range(self.dataset_num):
                if idx < self.lengths[i]:
                    return self.datasets[i][idx]
                idx -= self.lengths[i]
            raise IndexError("Index out of range")
        else:
            rand_val = random.random()
            dataset_idx = np.searchsorted(self.cumulative_ratios, rand_val)
            sample_idx = random.randint(0, self.lengths[dataset_idx] - 1)
            return self.datasets[dataset_idx][sample_idx]

