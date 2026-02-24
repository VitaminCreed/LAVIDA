# Code adapted from the anomalib library https://github.com/open-edge-platform/anomalib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from .base import ClipsIndexer, BaseVideoDataset
from pathlib import Path
from typing import Any
from ..data_utils import validate_path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import json
import os
from PIL import Image


ANOMALY_CLASS = ["bicycle", "fight", "throw", "pick", "hand truck", "run", "skate", "falling", "jumping",  "motorcycle", 
                 "tricycle", "cart", "waving stick", "hurdle", "vehicle", "play", "baby carriage", "truck"]



class ShanghaiTechDataset(BaseVideoDataset):
    """ShanghaiTech Dataset class.

    Args:
        split (Split): Dataset split - either ``Split.TRAIN`` or ``Split.TEST``
        root (Path | str): Path to the root directory containing the dataset.
            Defaults to ``"./datasets/shanghaitech"``.
        scene (int): Index of the dataset scene (category) in range [1, 13].
            Defaults to ``1``.
        clip_length_in_frames (int, optional): Number of frames in each video
            clip. Defaults to ``2``.
        frames_between_clips (int, optional): Number of frames between each
            consecutive video clip. Defaults to ``1``.
        target_frame (VideoTargetFrame): Specifies which frame in the clip to use
            for ground truth retrieval. Defaults to ``VideoTargetFrame.LAST``.
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.

    Example:
        >>> from anomalib.data.datasets import ShanghaiTechDataset
        >>> from anomalib.data.utils import Split
        >>> dataset = ShanghaiTechDataset(
        ...     root="./datasets/shanghaitech",
        ...     scene=1,
        ...     split=Split.TRAIN
        ... )
    """

    def __init__(
        self,
        data_root: Path | str='',
        list_root: Path | str='./lists/ShanghaiTech',
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.root = Path(data_root)
        self.list_root = Path(list_root)
        self.indexer_cls = ShanghaiTechTestClipsIndexer if self.inference else ShanghaiTechTrainClipsIndexer
        self.samples = make_shanghaitech_dataset(self.root, self.list_root, None, self.inference)


class ShanghaiTechTrainClipsIndexer(ClipsIndexer):
    """Clips indexer for ShanghaiTech training dataset.

    The train and test subsets use different file formats, so separate clips
    indexer implementations are needed.
    """

    @staticmethod
    def get_mask(
        idx: int,
        video_idx: int,
        frames_idx: int,
        video_shape: torch.Size|None = None,
    ) -> torch.Tensor | None:
        """No masks available for training set.

        Args:
            idx (int): Index of the clip.

        Returns:
            None: Training set has no masks.
        """
        del idx  # Unused argument
        return None
          


class ShanghaiTechTestClipsIndexer(ClipsIndexer):
    """Clips indexer for ShanghaiTech test dataset.

    The train and test subsets use different file formats, so separate clips
    indexer implementations are needed.
    """

    def get_mask(self, 
                 idx: int,
                 video_idx: int,
                 frames_idx: int,
                 video_shape: torch.Size|None = None,
    ) -> torch.Tensor | None:
        """Retrieve the masks from the file system.

        Args:
            idx (int): Index of the clip.

        Returns:
            torch.Tensor | None: Ground truth mask if available, else None.
        """
        video_idx, frames_idx = self.get_clip_location(idx)
        mask_file = self.mask_paths[video_idx]
        if mask_file == "":  # no gt masks available for this clip
            return None
        frames = self.clips[video_idx][frames_idx]

        vid_masks = np.load(mask_file)
        anomaly_masks=torch.tensor(np.take(vid_masks, frames, 0))
        gt_label = (anomaly_masks == 1).any(dim=(1, 2)).int()
        return {
            'anomaly_labels': None,
            'all_anomaly_types': ANOMALY_CLASS,
            'gt_label': gt_label ,
            'anomaly_masks':anomaly_masks[::self.sample_interval], 
        }

    def _compute_frame_pts(self) -> None:
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


def make_shanghaitech_dataset(root: Path, list_root: Path, scene: int | None = None, inference: bool | None = None) -> DataFrame:
    """Create ShanghaiTech dataset by parsing the file structure.

    The files are expected to follow the structure::

        root/
        ├── training/
        │   └── videos/
        │       ├── 01_001.avi
        │       └── ...
        └── testing/
            ├── frames/
            │   ├── 01_0014/
            │   │   ├── 000001.jpg
            │   │   └── ...
            │   └── ...
            └── test_pixel_mask/
                ├── 01_0014.npy
                └── ...

    Args:
        root (Path): Path to dataset root directory.
        scene (int | None, optional): Index of the dataset scene (category) in range [1, 13].
            If None, data from all scenes will be included. Defaults to None.
        inference (bool | None, optional): train or test. Defaults to None.

    Returns:
        DataFrame: DataFrame containing samples for the requested split and scene(s).

    """
    def get_scene_data(scene_prefix: str) -> DataFrame:
        """Helper function to get data for a specific scene."""
        train_root = root / "training/videos"
        train_list = [(str(train_root),) + filename.parts[-2:] for filename in train_root.glob(f"{scene_prefix}_*.avi")]
        train_samples = DataFrame(train_list, columns=["root", "folder", "image_path"])
        train_samples["inference"] = False

        test_root = Path(root) / "testing/frames"
        test_folders = [filename for filename in sorted(test_root.glob(f"{scene_prefix}_*")) if filename.is_dir()]
        test_folders = [folder for folder in test_folders if len(list(folder.glob("*.jpg"))) > 0]
        test_list = [(str(test_root),) + folder.parts[-2:] for folder in test_folders]
        test_samples = DataFrame(test_list, columns=["root", "folder", "image_path"])
        test_samples["inference"] = True
        samples = pd.concat([train_samples, test_samples], ignore_index=True)

        gt_root = Path(root) / "testing/test_pixel_mask"
        samples["mask_path"] = ""
        samples.loc[samples.root == str(test_root), "mask_path"] = (
            str(gt_root) + "/" + samples.image_path.str.split(".").str[0] + ".npy"
        )
        samples["image_path"] = samples.root + "/" + samples.image_path

        return samples


    root = validate_path(root)

    # if scene is None, process all scenes (1 to 13)
    if scene is None:
        all_samples = []
        for scene_num in range(1, 14):
            scene_prefix = str(scene_num).zfill(2)
            scene_samples = get_scene_data(scene_prefix)
            all_samples.append(scene_samples)
        samples = pd.concat(all_samples, ignore_index=True)
    else:
        # process a specific scene
        scene_prefix = str(scene).zfill(2)
        samples = get_scene_data(scene_prefix)

    # infer the task type
    samples.attrs["task"] = "classification" if (samples["mask_path"] == "").all() else "segmentation"

    # filter by inference if specified
    if inference is not None:
        samples = samples[samples.inference == inference]
        samples = samples.reset_index(drop=True)
        if inference:
            meta_file = list_root / 'ShanghaiTech_test_metadata.json'
        else:
            meta_file = list_root / 'ShanghaiTech_training_metadata.json'
        
        if os.path.exists(meta_file):
            metadata = pd.read_json(meta_file)
            if "image_path" not in metadata.columns and "video_paths" in metadata.columns:
                metadata.rename(columns={"video_paths": "image_path"}, inplace=True)
            samples = pd.merge(samples, metadata, on="image_path", how="inner")

    return samples