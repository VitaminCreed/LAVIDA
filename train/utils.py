import numpy as np
import torch
import torch.distributed as dist
from enum import Enum
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def average_loss_dicts(loss_dicts):
    if not loss_dicts:
        return {}

    total_loss_dict = defaultdict(float)
    count = len(loss_dicts)
    for loss_dict in loss_dicts:
        for key, value in loss_dict.items():
            total_loss_dict[key] += value
    avg_loss_dict = {key: total_loss / count for key, total_loss in total_loss_dict.items()}
    return avg_loss_dict


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)
    
@dataclass
class VideoData:
    pred_scores: List[np.ndarray] = field(default_factory=list)
    gt_labels: List[np.ndarray] = field(default_factory=list)


class VideoResultSummary:
    def __init__(self, 
                 apply_sigmoid: bool = False,
                 sigma: float = 15.0,
                 metric: str = 'AUC',
        ):
        self.results = defaultdict(VideoData)
        self.apply_sigmoid = apply_sigmoid
        self.sigma = sigma
        self._is_aggregated = False
        self.metric = metric
        assert metric in ['AUC', 'AP']

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

        return tensor.detach().cpu().numpy()

    def update(self, 
               video_names: List[str], 
               pred_scores: torch.Tensor, 
               gt_labels: torch.Tensor) -> None:
        pred_scores_np = self._to_numpy(pred_scores)
        gt_labels_np = self._to_numpy(gt_labels)

        for idx, name in enumerate(video_names):
            self.results[name].pred_scores.append(pred_scores_np[idx])
            self.results[name].gt_labels.append(gt_labels_np[idx])

    def _aggregate_video_data(self, video_data: VideoData) -> None:
        video_data.pred_scores = [np.concatenate(video_data.pred_scores)]
        video_data.gt_labels = [np.concatenate(video_data.gt_labels)]


    def aggregate_results(self) -> None:
        for video_data in self.results.values():
            self._aggregate_video_data(video_data)
        self._is_aggregated = True

    def _process_scores(self, scores: np.ndarray) -> np.ndarray:
        if self.apply_sigmoid:
            scores =  1 / (1 + np.exp(-scores))
        return scores

    @staticmethod
    def _compute_extended_score(scores: np.ndarray, labels: np.ndarray, metric) -> float:
        extended_scores = np.concatenate(([0], scores, [1]))
        extended_labels = np.concatenate(([0], labels, [1]))
        if metric == 'AUC':
            return roc_auc_score(extended_labels, extended_scores)
        elif metric == 'AP':
            return average_precision_score(extended_labels, extended_scores)

    def calculate_score(self, output_path: str = 'result.npy') -> Tuple[float, float]:
        self.aggregate_results()

        all_scores, all_labels = [], []
        video_score_list = []

        for video_data in self.results.values():
            processed_scores = self._process_scores(video_data.pred_scores[0])

            video_auc = self._compute_extended_score(
                processed_scores, 
                video_data.gt_labels[0],
                self.metric
            )
            video_score_list.append(video_auc)
            all_scores.append(processed_scores)
            all_labels.append(video_data.gt_labels[0])

        global_scores = np.concatenate(all_scores)
        global_labels = np.concatenate(all_labels)
        np.save(output_path, np.column_stack((global_scores, global_labels)))
        frame_score = self._compute_extended_score(global_scores, global_labels, self.metric)
        return frame_score, np.mean(video_score_list)
        

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
