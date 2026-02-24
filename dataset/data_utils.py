# Code adapted from the anomalib library https://github.com/open-edge-platform/anomalib
import torch
from pathlib import Path
import torch
from torchvision.transforms.v2.functional import to_dtype, to_image
from PIL import Image
import numpy as np
from torchvision.tv_tensors import Mask
import re
import os
import random


def read_image(path: str | Path, as_tensor: bool = False) -> torch.Tensor | np.ndarray:
    """Read RGB image from disk.

    Args:
        path (str | Path): Path to image file
        as_tensor (bool): If ``True``, return torch.Tensor. Defaults to ``False``

    Returns:
        torch.Tensor | np.ndarray: Image as tensor or array, normalized to [0,1]

    Examples:
        >>> image = read_image("image.jpg")
        >>> type(image)
        <class 'numpy.ndarray'>

        >>> image = read_image("image.jpg", as_tensor=True)
        >>> type(image)
        <class 'torch.Tensor'>
    """
    image = Image.open(path).convert("RGB")
    return to_dtype(to_image(image), torch.float32, scale=True) if as_tensor else np.array(image) / 255.0


def read_mask(path: str | Path, as_tensor: bool = False) -> torch.Tensor | np.ndarray:
    """Read grayscale mask from disk.

    Args:
        path (str | Path): Path to mask file
        as_tensor (bool): If ``True``, return torch.Tensor. Defaults to ``False``

    Returns:
        torch.Tensor | np.ndarray: Mask as tensor or array

    Examples:
        >>> mask = read_mask("mask.png")
        >>> type(mask)
        <class 'numpy.ndarray'>

        >>> mask = read_mask("mask.png", as_tensor=True)
        >>> type(mask)
        <class 'torch.Tensor'>
    """
    image = Image.open(path).convert("L")
    return Mask(to_image(image).squeeze() / 255, dtype=torch.uint8) if as_tensor else np.array(image)

def validate_path(
    path: str | Path,
    base_dir: str | Path | None = None,
    should_exist: bool = True,
    extensions: tuple[str, ...] | None = None,
) -> Path:
    """Validate path for existence, permissions and extension.

    Args:
        path: Path to validate
        base_dir: Base directory to restrict file access
        should_exist: If ``True``, verify path exists
        extensions: Allowed file extensions

    Returns:
        Validated Path object

    Raises:
        TypeError: If path is invalid type
        ValueError: If path is too long or has invalid characters/extension
        FileNotFoundError: If path doesn't exist when required
        PermissionError: If path lacks required permissions

    Example:
        >>> path = validate_path("./datasets/image.png", extensions=(".png",))
        >>> path.suffix
        '.png'
    """
    # Check if the path is of an appropriate type
    if not isinstance(path, str | Path):
        raise TypeError("Expected str, bytes or os.PathLike object, not " + type(path).__name__)

    # Check if the path is too long
    if len(str(path)) > 512:
        msg = f"Path is too long: {path}"
        raise ValueError(msg)

    # Check if the path contains non-printable characters
    if not re.compile(r"^[\x20-\x7E]+$").match(str(path)):
        msg = f"Path contains non-printable characters: {path}"
        raise ValueError(msg)

    # Sanitize paths
    path = Path(path).resolve()
    base_dir = Path(base_dir).resolve() if base_dir else Path.home()

    # In case path ``should_exist``, the path is valid, and should be
    # checked for read and execute permissions.
    if should_exist:
        # Check if the path exists
        if not path.exists():
            msg = f"Path does not exist: {path}"
            raise FileNotFoundError(msg)

        # Check the read and execute permissions
        if not (os.access(path, os.R_OK) or os.access(path, os.X_OK)):
            msg = f"Read or execute permissions denied for the path: {path}"
            raise PermissionError(msg)

    # Check if the path has one of the accepted extensions
    if extensions is not None and path.suffix not in extensions:
        msg = f"Path extension is not accepted. Accepted: {extensions}. Path: {path}"
        raise ValueError(msg)

    return path


def select_anomalies(
    anomaly_classes, 
    all_classes, 
    max_n_anomalies, 
    is_anomaly, 
    one_true_anomaly=False,
    random_sample=True,
):
    """Selects and combines anomaly/normal classes according to specified sampling rules.
    
    Performs intelligent sampling of anomaly classes while maintaining:
    - No duplicate classes in output
    - Minimum 1 selected class guarantee
    - Configurable sampling behavior
    
    Args:
        anomaly_classes: List of known anomaly classes (may contain duplicates)
        all_classes: Complete list of available classes (may contain duplicates)
        max_n_anomalies: Maximum number of classes to select (must be ≥1)
        is_anomaly: Whether to shuffle and label results (True) or return only normal classes (False)
        one_true_anomaly: When True, selects exactly 1 true anomaly plus supplemental normal classes
        random_sample: When True, samples random count (1-max_n_anomalies); when False uses max_n_anomalies
    
    Returns:
        Tuple containing:
        - selected_labels: Binary labels (1=anomaly, 0=normal)
        - selected_classes: Corresponding class names
        - [only when one_true_anomaly=True] The single true anomaly class
    
    Raises:
        ValueError: If input constraints are violated
    """
    if all_classes==None or len(all_classes)==0:
        raise ValueError("all_classes must not be empty")
    if max_n_anomalies < 1:
        raise ValueError("max_n_anomalies must ≥ 1")

    unique_anomaly_set = set(anomaly_classes)
    all_classes_set = set(all_classes)
    other_classes_set = all_classes_set - unique_anomaly_set

    if one_true_anomaly and unique_anomaly_set:
        selected_anomaly = [random.choice(list(unique_anomaly_set))]
        remaining_slots = max(0, max_n_anomalies - 1)
    else:
        selected_anomaly = list(unique_anomaly_set)
        remaining_slots = max(0, max_n_anomalies - len(unique_anomaly_set))

    selected_other = []
    if remaining_slots > 0 and other_classes_set:
        if random_sample:
            max_to_select = min(remaining_slots, len(other_classes_set))
            min_to_select = 1 
            if is_anomaly:
                n_to_select = random.randint(min_to_select, max_to_select)
            else:
                n_to_select = random.randint(min_to_select, max_to_select)
        else:
            n_to_select = min(remaining_slots, len(other_classes_set))
        selected_other = random.sample(list(other_classes_set), n_to_select)

    if is_anomaly:
        selected_anomalies = selected_anomaly + selected_other
        selected_labels = [1] * len(selected_anomaly) + [0] * len(selected_other)
        combined = list(zip(selected_anomalies, selected_labels))
        random.shuffle(combined)
        selected_anomalies, selected_labels = zip(*combined)
        selected_anomalies = list(selected_anomalies)
        selected_labels = list(selected_labels)
    else:
        selected_anomalies = selected_other
        selected_labels = [0] * len(selected_other)
        
    assert selected_anomalies != None and len(selected_anomalies) > 0, "selected_anomalies must ≥1"
    if one_true_anomaly:
        return selected_labels, selected_anomalies, selected_anomaly
    else:
        return selected_labels, selected_anomalies


