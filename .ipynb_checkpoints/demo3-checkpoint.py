# 输入：模型存放地址，视频地址，起始帧，结束帧，输出地址
# 输出：图像
from model import LavidaMetaModel
import torch
from torch.amp import autocast
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import functional
from torchvision.transforms import InterpolationMode
import decord
import cv2
from pathlib import Path
import numpy as np
from qwen_vl_utils.vision_process import smart_resize, VIDEO_MIN_PIXELS, VIDEO_TOTAL_PIXELS, VIDEO_MAX_PIXELS, FRAME_FACTOR, IMAGE_FACTOR
from model import LavidaForCausalLM
import transformers
from peft import LoraConfig, get_peft_model
from train import YamlArgs
import argparse
import deepspeed
from dataset.messages_template import create_structured_template
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, Qwen2VLConfig, Qwen2VLForConditionalGeneration
import os
from typing import Iterator, Optional
from tqdm import tqdm
from typing import List, Tuple
import tifffile
from pathlib import Path
AutoModelForCausalLM.register(config_class=Qwen2VLConfig, model_class=Qwen2VLForConditionalGeneration)


def load_args(args_file='./train/train_para.yaml'):
    yaml_to_args = YamlArgs(args_file)
    parser = argparse.ArgumentParser(description="Training script")
    parser = yaml_to_args.add_to_parser(parser)
    args = parser.parse_args()
    return args


def extract_video_segments(
    video_path: str,
    clip_len: int = 16,
    sample_interval: int = 2,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    pad_last: bool = False,
) -> Iterator[torch.Tensor]:
    """
    视频片段生成器（支持区间限定采样）
    
    Args:
        video_path: 视频路径
        clip_len: 每个片段包含的帧数 (默认16)
        sample_interval: 采样间隔帧数 (默认2)
        start_frame: 起始帧位置 (默认0)
        end_frame: 结束帧位置 (默认None表示视频末尾)
        pad_last: 是否对最后一个不完整片段补零 (默认False)
    
    Returns:
        迭代器返回视频片段Tensor [T,C,H,W]
    """
    # 参数校验
    assert clip_len > 0, "clip_len必须大于0"
    assert sample_interval > 0, "sample_interval必须大于0"
    
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    end_frame = total_frames if end_frame is None else min(end_frame, total_frames)
    start_frame = max(0, start_frame)
    
    # 计算每个片段需要的原始帧跨度
    span_frames = (clip_len - 1) * sample_interval + 1
    
    current_frame = start_frame
    while current_frame < end_frame:
        # 计算实际结束位置
        segment_end = min(current_frame + span_frames, end_frame)
        
        # 生成采样帧索引
        frame_indices = range(
            current_frame,
            segment_end,
            sample_interval
        )
        
        # 提取帧数据
        frames = vr.get_batch(frame_indices).asnumpy()  # [T,H,W,C]
        frames = torch.from_numpy(frames.transpose(0, 3, 1, 2))  # [T,C,H,W]
        # 处理不完整片段
        if frames.shape[0] < clip_len:
            if pad_last:
                padding = torch.zeros(
                    (clip_len - frames.shape[0], *frames.shape[1:]),
                    dtype=frames.dtype
                )
                frames = torch.cat([frames, padding], dim=0)
            else:
                current_frame += span_frames
                continue
        
        yield frames
        current_frame += span_frames
        
        
def extract_video_segments_shanghaitech(
    video_path: str,
    clip_len: int = 16,
    sample_interval: int = 2,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    pad_last: bool = False,
) -> Iterator[torch.Tensor]:
    """
    视频片段生成器（支持区间限定采样）
    
    Args:
        video_path: 视频路径
        clip_len: 每个片段包含的帧数 (默认16)
        sample_interval: 采样间隔帧数 (默认2)
        start_frame: 起始帧位置 (默认0)
        end_frame: 结束帧位置 (默认None表示视频末尾)
        pad_last: 是否对最后一个不完整片段补零 (默认False)
    
    Returns:
        迭代器返回视频片段Tensor [T,C,H,W]
    """
    # 参数校验
    assert clip_len > 0, "clip_len必须大于0"
    assert sample_interval > 0, "sample_interval必须大于0"
    
    vr  = sorted(Path(video_path).glob("*.jpg"))
    
    total_frames = len(vr)
    end_frame = total_frames if end_frame is None else min(end_frame, total_frames)
    start_frame = max(0, start_frame)
    
    # 计算每个片段需要的原始帧跨度
    span_frames = (clip_len - 1) * sample_interval + 1
    
    current_frame = start_frame
    while current_frame < end_frame:
        # 计算实际结束位置
        segment_end = min(current_frame + span_frames, end_frame)
        
        # 生成采样帧索引
        frame_indices = range(
            current_frame,
            segment_end,
            sample_interval
        )
        
        # 提取帧数据
        frame_paths = [vr[i] for i in frame_indices] # [T,H,W,C]
        frames = []
        for frame_path in frame_paths:
            frame = Image.open(frame_path).convert("RGB")
            frame_tensor = torch.from_numpy(np.array(frame)).permute(2, 0, 1)  # HWC -> CHW
            frames.append(frame_tensor)
        frames = torch.stack(frames)
        # frames = vr.get_batch(frame_indices).asnumpy() # [T,C,H,W]
        # 处理不完整片段
        if frames.shape[0] < clip_len:
            if pad_last:
                padding = torch.zeros(
                    (clip_len - frames.shape[0], *frames.shape[1:]),
                    dtype=frames.dtype
                )
                frames = torch.cat([frames, padding], dim=0)
            else:
                current_frame += span_frames
                continue
        
        yield frames
        current_frame += span_frames
        
        
def extract_video_segments_ucsdped(
    video_path: str,
    clip_len: int = 16,
    sample_interval: int = 2,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    pad_last: bool = False,
) -> Iterator[torch.Tensor]:
    """
    视频片段生成器（支持区间限定采样）
    
    Args:
        video_path: 视频路径
        clip_len: 每个片段包含的帧数 (默认16)
        sample_interval: 采样间隔帧数 (默认2)
        start_frame: 起始帧位置 (默认0)
        end_frame: 结束帧位置 (默认None表示视频末尾)
        pad_last: 是否对最后一个不完整片段补零 (默认False)
    
    Returns:
        迭代器返回视频片段Tensor [T,C,H,W]
    """
    # 参数校验
    assert clip_len > 0, "clip_len必须大于0"
    assert sample_interval > 0, "sample_interval必须大于0"
    
    vr  = sorted(Path(video_path).glob("*.tif"))
    
    total_frames = len(vr)
    end_frame = total_frames if end_frame is None else min(end_frame, total_frames)
    start_frame = max(0, start_frame)
    
    # 计算每个片段需要的原始帧跨度
    span_frames = (clip_len - 1) * sample_interval + 1
    
    current_frame = start_frame
    while current_frame < end_frame:
        # 计算实际结束位置
        segment_end = min(current_frame + span_frames, end_frame)
        
        # 生成采样帧索引
        frame_indices = range(
            current_frame,
            segment_end,
            sample_interval
        )
        
        # 提取帧数据
        frame_paths = [vr[i] for i in frame_indices] # [T,H,W,C]
        
        frames = []
        for frame_path in frame_paths:
            try:
                img = tifffile.imread(str(frame_path))
            except:
                return None

            img_array = np.array(img)
        
            # 处理不同通道数的图像
            if img_array.ndim == 2:  # 灰度图 (H, W)
                img_array = np.expand_dims(img_array, axis=-1)  # 添加通道维度 -> (H, W, 1)
            elif img_array.ndim == 3 and img_array.shape[2] > 3:  # 多通道图像
                # 只取前3个通道 (RGB)
                img_array = img_array[:, :, :3]
            
            frame_tensor = torch.from_numpy(img_array).permute(2, 0, 1).repeat(3, 1, 1)
            frames.append(frame_tensor)
        frames = torch.stack(frames)
        # frames = vr.get_batch(frame_indices).asnumpy() # [T,C,H,W]
        # 处理不完整片段
        if frames.shape[0] < clip_len:
            if pad_last:
                padding = torch.zeros(
                    (clip_len - frames.shape[0], *frames.shape[1:]),
                    dtype=frames.dtype
                )
                frames = torch.cat([frames, padding], dim=0)
            else:
                current_frame += span_frames
                continue
        
        yield frames
        current_frame += span_frames

def load_anomaly_tracks(mask_folder, video_name):
    is_abnormal = video_name.startswith('abnormal')
    anomaly_tracks = []

    if is_abnormal:
        tracks_path = Path(mask_folder) / f"{video_name}_tracks.txt"

        if tracks_path.exists():
            with tracks_path.open('r') as f:
                anomaly_tracks = [
                    list(map(lambda x: int(float(x)), line.strip().split(',')))  
                    for line in f if line.strip() 
                ]
    return anomaly_tracks

def get_masks_ubnormal(video_path, start_idx, end_idx, mask_folder, video_shape=None, anomaly_tracks=None):
    video_name = Path(video_path).stem
    frames = list(range(start_idx, end_idx))

    all_mask_list = []
    anomaly_mask_list = []

    for frame_idx in frames:
        mask_file_name = f"{video_name}_{frame_idx:04d}_gt.png"
        mask_file_path = Path(mask_folder) / mask_file_name

        mask = cv2.imread(str(mask_file_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask file not found: {mask_file_path}")

        mask = torch.from_numpy(mask).to(torch.uint8)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        if video_shape is not None and mask.shape != video_shape:
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),  # 增加 batch 和 channel 维度
                size=video_shape,
                mode='nearest'
            ).squeeze(0).squeeze(0).to(torch.uint8)

        # 生成 all_mask
        all_mask = (mask > 0).to(torch.uint8)

        anomaly_tracks = load_anomaly_tracks(mask_folder, video_name)
        if anomaly_tracks is None:
            anomaly_mask = torch.zeros_like(mask, dtype=torch.uint8)
        else:
            anomaly_mask = torch.zeros_like(mask, dtype=torch.uint8)
            for track in anomaly_tracks:
                obj_id, start_frame, end_frame = track
                if start_frame <= frame_idx <= end_frame:
                    anomaly_mask[mask == obj_id] = 1

        all_mask_list.append(all_mask)
        anomaly_mask_list.append(anomaly_mask)

    all_masks = torch.stack(all_mask_list, dim=0)  # [T, H, W]
    anomaly_masks = torch.stack(anomaly_mask_list, dim=0)  # [T, H, W]
    normal_masks = all_masks - anomaly_masks  # [T, H, W]
    return all_masks, anomaly_masks, normal_masks

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


def process_pred_masks(pred_masks, target_shape):
    T, H_pred, W_pred = pred_masks.shape
    H, W = target_shape

    # 调整预测掩码大小
    processed_masks = torch.nn.functional.interpolate(
        pred_masks.unsqueeze(1).float(),  # 增加 channel 维度 [T, 1, H_pred, W_pred]
        size=(H, W),
        mode='nearest'
    ).squeeze(1)  # 移除 channel 维度 [T, H, W]

    # 将大于 0 的部分设置为掩码区域（值为 1），小于等于 0 的部分设置为非掩码区域（值为 0）
    processed_masks = (processed_masks > 0).to(torch.uint8)

    return processed_masks



import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

def visualize_video_with_mask(video_tensor, mask_tensor, save_path, alpha=0.3):
    """
    将视频片段和mask显示在一张图像中，所有帧显示在一行
    
    Args:
        video_tensor: torch.Tensor, shape [T, 3, H, W], 视频片段
        mask_tensor: torch.Tensor, shape [T, 1, 1024, 1024], mask
        save_path: str, 保存路径
        alpha: float, mask透明度 (0-1)
    """
    
    # 确保输入是torch tensor
    if not isinstance(video_tensor, torch.Tensor):
        video_tensor = torch.tensor(video_tensor)
    if not isinstance(mask_tensor, torch.Tensor):
        mask_tensor = torch.tensor(mask_tensor)
    
    # 获取视频参数
    T, C, H, W = video_tensor.shape
    T_mask, _, mask_H, mask_W = mask_tensor.shape
    
    # 确保时间维度一致
    T = min(T, T_mask)
    
    # 将tensor转换为numpy并调整维度顺序 [T, H, W, C]
    video_np = video_tensor[:T].permute(0, 2, 3, 1).cpu().numpy()
    mask_np = mask_tensor[:T, 0].cpu().numpy()  # [T, 1024, 1024]
    
    # 检测数据类型和范围，决定是否需要处理
    if video_np.dtype in [np.uint8, np.int32, np.int64]:
        # 整数类型，检查范围
        if video_np.max() <= 255:
            video_display = video_np.astype(np.uint8)
        else:
            # 如果超过255，可能需要缩放
            video_display = np.clip(video_np, 0, 255).astype(np.uint8)
    else:
        # 浮点类型
        if video_np.max() <= 1.0:
            video_display = video_np  # 已经在0-1范围
        else:
            video_display = np.clip(video_np / 255.0, 0, 1)  # 假设是0-255范围
    
    # 处理mask - 通常mask是二值的或者在0-1范围
    if mask_np.dtype in [np.bool_, bool]:
        mask_display = mask_np.astype(np.float32)
    elif mask_np.dtype in [np.uint8, np.int32, np.int64]:
        if mask_np.max() <= 1:
            mask_display = mask_np.astype(np.float32)
        else:
            mask_display = (mask_np / 255.0).astype(np.float32)
    else:
        mask_display = mask_np.astype(np.float32)
        if mask_display.max() > 1.0:
            mask_display = mask_display / 255.0
    
    # 如果视频尺寸和mask尺寸不一致，将mask resize到视频尺寸
    if H != mask_H or W != mask_W:
        resized_masks = []
        for i in range(T):
            resized_mask = cv2.resize(mask_display[i], (W, H), interpolation=cv2.INTER_LINEAR)
            resized_masks.append(resized_mask)
        mask_display = np.array(resized_masks)
    
    # 创建图像显示：2行T列，第一行显示原始帧，第二行显示叠加帧
    fig, axes = plt.subplots(2, T, figsize=(T * 3, 6))
    
    # 如果只有一帧，确保axes是2D数组
    if T == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(T):
        # 原始视频帧 (第一行)
        axes[0, i].imshow(video_display[i])
        axes[0, i].set_title(f'Frame {i+1} - Original')
        axes[0, i].axis('off')
        
        # 创建叠加图像
        # 确保video_display是浮点格式以便进行混合运算
        if video_display.dtype == np.uint8:
            overlay_base = video_display[i].astype(np.float32) / 255.0
        else:
            overlay_base = video_display[i].copy()
        
        # 创建红色mask
        red_mask = np.zeros_like(overlay_base)
        red_mask[:, :, 1] = mask_display[i]  # 
        
        # 应用透明度叠加
        overlay_img = overlay_base * (1 - alpha * mask_display[i][:, :, np.newaxis]) + \
                     red_mask * alpha
        
        overlay_img = np.clip(overlay_img, 0, 1)
        
        # 显示叠加后的图像 (第二行)
        axes[1, i].imshow(overlay_img)
        axes[1, i].set_title(f'Frame {i+1} - With Mask')
        axes[1, i].axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"图像已保存到: {save_path}")
    
    
plt.rcParams.update({
    'font.family': 'serif',           # 使用serif字体（更学术化）
    # 'font.serif': ['Times New Roman', 'Computer Modern Roman'],
    'font.size': 11,                  # 基础字体大小
    'axes.labelsize': 12,             # 坐标轴标签字体大小
    'axes.titlesize': 13,             # 标题字体大小
    'xtick.labelsize': 16,            # x轴刻度标签字体大小
    'ytick.labelsize': 16,            # y轴刻度标签字体大小
    'legend.fontsize': 16,            # 图例字体大小
    'figure.titlesize': 16,           # 图形标题字体大小
    'axes.linewidth': 2,              # 坐标轴线宽
    'grid.linewidth': 0.8,            # 网格线宽
    'lines.linewidth': 2.5,           # 线条宽度
    'patch.linewidth': 0.5,           # 填充区域边框宽度
})

def plot_pred_score_with_windows(pred_score: torch.Tensor, 
                                time_windows: List[List[int]], 
                                title: str = "Prediction Scores Over Time",
                                figsize: Tuple[int, int] = (30, 4),
                                line_color: str = '#2E86C1',
                                window_color: str = '#F8C8DC',
                                window_alpha: float = 0.4,
                                save_path: str = 'scores.png') -> plt.Figure:
    # Convert torch tensor to numpy array
    if isinstance(pred_score, torch.Tensor):
        pred_score_np = pred_score.detach().cpu().numpy()
    else:
        pred_score_np = np.array(pred_score)
    
    # Create time step indices
    time_steps = np.arange(len(pred_score_np))
    
    # Create figure and axis with academic style
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    # Set white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # First, plot time window backgrounds
    for i, window in enumerate(time_windows):
        start_idx, end_idx = window
        # Ensure indices are within valid range
        start_idx = max(0, min(start_idx, len(pred_score_np) - 1))
        end_idx = max(start_idx, min(end_idx, len(pred_score_np) - 1))
        
        # Add background color regions
        ax.axvspan(start_idx, end_idx, color=window_color, alpha=window_alpha, 
                  label='Anomaly Windows' if i == 0 else "", zorder=1)
    
    # Plot prediction score line
    line = ax.plot(time_steps, pred_score_np, color=line_color, linewidth=2, 
                   label='Prediction Scores', zorder=3)
    
    # Add subtle markers only at window boundaries for clarity
    for window in time_windows:
        start_idx, end_idx = window
        start_idx = max(0, min(start_idx, len(pred_score_np) - 1))
        end_idx = max(start_idx, min(end_idx, len(pred_score_np) - 1))
        ax.plot(start_idx, pred_score_np[start_idx], 'o', color=line_color, 
                markersize=4, zorder=4)
        ax.plot(end_idx, pred_score_np[end_idx], 'o', color=line_color, 
                markersize=4, zorder=4)
    
    # Set labels with proper academic formatting
    ax.set_xlabel('Time Step', fontweight='normal')
    ax.set_ylabel('Prediction Score', fontweight='normal')
    ax.set_title(title, fontweight='bold', pad=15)
    
    # Academic-style grid
    ax.grid(True, linestyle='-', alpha=0.3, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    
    # Professional legend positioning
    legend = ax.legend(loc='upper right', frameon=True, fancybox=False, 
                      shadow=False, framealpha=0.9, edgecolor='black')
    legend.get_frame().set_linewidth(0.8)
    
    # Set axis limits with small margins
    ax.set_xlim(-0.5, len(pred_score_np) - 0.5)
    y_margin = (np.max(pred_score_np) - np.min(pred_score_np)) * 0.05
    ax.set_ylim(np.min(pred_score_np) - y_margin, np.max(pred_score_np) + y_margin)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    # Adjust layout for academic papers
    plt.tight_layout(pad=1.0)
    
    # Save the figure with high quality for academic use
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='png')
    
    print(f"Figure saved to: {save_path}")
    
    return fig


@torch.inference_mode()
def demo(model, tokenizer, processor):
    model_cfg =  "configs/sam2.1/sam2.1_hiera_b+.yaml"
    # anomaly_types = ['fighting', 'shooting', 'riot', 'abuse', 'car accident', 'explosion']
    
    device = 0
    # start_frame = 0
    # end_frame = 300
    # clip_len = 8
    # llm_sample_frames = 8
    # sample_interval = 5
    # video_path = "/root/autodl-tmp/datasets/XD-Violence/XD-Violence-test/videos/Quantum.Of.Solace.2008__#01-30-26_01-30-42_label_B2-B1-0.mp4"
    # # annotation_dir = "/root/autodl-tmp/datasets/UBnormal/Scene9/abnormal_scene_9_scenario_4_annotations"
    
    anomaly_types = ['running', 'jumping','falling', 'fighting', 'sleeping', 'crawling', 'having a seizure', 
                 'laying down', 'dancing', 'stealing', 'rotating 360 degrees', 'shuffling', 'injured', 
                 'drunk', 'stumbling walk', 'car accident', 'fire', 'smoke', 'jaywalking', 'driving outside lane']
    start_frame = 40
    end_frame = 120
    clip_len = 8
    llm_sample_frames = 8
    sample_interval = 2
    anomaly_windows = []
    video_path = "/root/autodl-tmp/datasets/UBnormal/Scene1/abnormal_scene_1_scenario_1.mp4"
    # annotation_dir = "/root/autodl-tmp/datasets/UBnormal/Scene9/abnormal_scene_9_scenario_4_annotations"
    
    # anomaly_types = ['running']
    # start_frame = 20
    # end_frame = 130
    # anomaly_windows = [[40, 110]]
    # anomaly_windows = np.array(anomaly_windows) - start_frame
    # clip_len = 2
    # llm_sample_frames = 2
    # sample_interval = 1
    # num_clip = (end_frame - start_frame) // clip_len // sample_interval
    # video_path = "/root/autodl-tmp/datasets/UBnormal/Scene1/abnormal_scene_1_scenario_1.mp4"
    # annotation_dir = "/root/autodl-tmp/datasets/UBnormal/Scene1/abnormal_scene_1_scenario_1_annotations"
    
    # anomaly_types = ['fighting']
    # start_frame = 0
    # end_frame = 300
    # anomaly_windows = [[136, 280]]
    # anomaly_windows = np.array(anomaly_windows) - start_frame
    # clip_len = 8
    # llm_sample_frames = 8
    # sample_interval = 1
    # num_clip = (end_frame - start_frame) // clip_len // sample_interval
    # video_path = "/root/autodl-tmp/datasets/XD-Violence/XD-Violence-test/videos/Mission.Impossible.II.2000__#01-29-30_01-29-44_label_B1-0-0.mp4"
    # annotation_dir = "/root/autodl-tmp/datasets/UBnormal/Scene1/abnormal_scene_1_scenario_1_annotations"
    
    
    # anomaly_types = ['throwing']
    # start_frame = 200
    # end_frame = 250
    # anomaly_windows = []
    # anomaly_windows = np.array(anomaly_windows) - start_frame
    # clip_len = 8
    # llm_sample_frames = 8
    # sample_interval = 4
    # num_clip = (end_frame - start_frame) // clip_len // sample_interval
    # video_path = "/root/autodl-tmp/datasets/shanghaitech/testing/frames/03_0036"
    
    
    # anomaly_types = ["car", "cart", "bicycle"]
    # start_frame = 90
    # end_frame = 180
    # anomaly_windows = [[0, 180]]
    # anomaly_windows = np.array(anomaly_windows) - start_frame
    # clip_len = 4
    # llm_sample_frames = 4
    # sample_interval = 1
    # num_clip = (end_frame - start_frame) // clip_len // sample_interval
    # video_path = "/root/autodl-tmp/datasets/UCSDped2/UCSDped2/Test/Test004"
    
    # anomaly_types = ['explosion', 'fire']
    # start_frame = 0
    # end_frame = 70
    # anomaly_windows = []
    # anomaly_windows = np.array(anomaly_windows) - start_frame
    # clip_len = 8
    # llm_sample_frames = 8
    # sample_interval = 2
    # video_path = "/root/autodl-tmp/datasets/XD-Violence/XD-Violence-test/videos/v=bhZs3ALdL7Y__#1_label_G-0-0.mp4"
    
    
    # anomaly_types = ['fire']
    # start_frame = 0
    # end_frame = 90
    # anomaly_windows = []
    # anomaly_windows = np.array(anomaly_windows) - start_frame
    # clip_len = 8
    # llm_sample_frames = 8
    # sample_interval = 2
    # video_path = "/root/autodl-tmp/datasets/MeVis/train/JPEGImages/0a860edc9877"
    

    num_clip = (end_frame - start_frame) // clip_len // sample_interval
    ckpt_pth = "./ckpt/checkpoint.pt"
    clip_ckpt = "/root/autodl-tmp/LLM_ckpt/clip-vit-base-patch32"
    clip_tokenizer = transformers.AutoTokenizer.from_pretrained(clip_ckpt)
    # model = AutoModelForCausalLM.from_pretrained(
    if model is None:
        model = LavidaForCausalLM.from_pretrained(
            ckpt_pth,
            device_map={"": device}, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
            output_loading_info=False,
            # device_map=None,
            # precision="bf16",
        )
    model.eval()
    # tokenizer = AutoTokenizer.from_pretrained(ckpt_pth)
    if processor is None:
        processor = AutoProcessor.from_pretrained(ckpt_pth)

    sam2_transform = transforms.Compose([
        transforms.Resize((1024, 1024)), 
        transforms.ConvertImageDtype(torch.float32), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
    ])
    
    # vr = decord.VideoReader(video_path)
    # video_shape = vr[0].shape[:2]
    # assert end_frame < len(vr), len(vr)
    
    
    video, pred_masks, targ_masks, pred_scores, target_labels = [], [], [], [], []
    seg_token_logits, seg_loc_logits, output_token_logits = [], [], []
    for video_clip in tqdm(extract_video_segments(video_path, clip_len, sample_interval, start_frame, end_frame), total=num_clip):
        aug_video_clip = sam2_transform(video_clip)
        
        nframes = video_clip.shape[0]
        all_frame_idx = np.arange(nframes) 
        cond_frame_idx = np.linspace(0, len(all_frame_idx)-1, num=llm_sample_frames, dtype=np.int32) 
        frame_idxs = {
            'all_frame_idx': all_frame_idx.tolist(), 
            'cond_frame_idx': cond_frame_idx.tolist(), 
            'non_cond_frame_idx': np.setdiff1d(all_frame_idx, cond_frame_idx).tolist()
        }
        
        pixel_values_videos = qwen_resize_video(
            video=video_clip[frame_idxs['cond_frame_idx']],
            min_pixels=256*28*28,  # 128
            max_pixels=512*28*28,  # 768
        )
        
        message = create_structured_template(
            path='', 
            type='video', 
            nframes=nframes,
            anomaly_list=anomaly_types,
        )
        
        text = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
        anomaly_token = clip_tokenizer(anomaly_types, padding=True, return_tensors="pt")

        conversations = processor(
            text=text,
            images=None,
            videos=[pixel_values_videos],  # 保留list形式
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        input_ids = conversations['input_ids']
        pixel_values_videos = conversations['pixel_values_videos']
        attention_masks = conversations['attention_mask']
        video_grid_thw = conversations['video_grid_thw']   
        input_ids_lists = input_ids.tolist()
        
        with torch.no_grad() and torch.autocast(device_type=f'cuda:{device}', dtype=torch.bfloat16):
            output_dict = model(
                input_ids=input_ids.cuda(), 
                images=[aug_video_clip.cuda()],
                pixel_values=None,
                pixel_values_videos=pixel_values_videos.cuda(), 
                frame_idx=[frame_idxs],
                anomaly_types=[anomaly_token],
                attention_mask=attention_masks.cuda(), 
                image_grid_thw=None,
                video_grid_thw=video_grid_thw.cuda(),  
                labels=None,  
                anomaly_labels=None,
                target_masks=None,  
                target_labels=None,
                inference=True,  
            )
            
        pred_mask = output_dict["pred_masks"].cpu()
        pred_score = output_dict["pred_scores"].cpu()
        pred_mask = (pred_mask > 0).int()
        pred_masks.append(pred_mask)
        pred_scores.append(pred_score)
        video.append(video_clip)
        seg_token_logits.append(output_dict["seg_token_logits"].cpu())
        seg_loc_logits.append(output_dict["seg_loc_logits"].cpu())
        output_token_logits.append(output_dict["output_token_logits"].cpu())
        
        
    
    pred_masks = torch.cat(pred_masks, dim=0)
    pred_scores = torch.cat(pred_scores, dim=0).to(torch.float32).cpu()
    pred_scores = torch.repeat_interleave(pred_scores, sample_interval)
    
    seg_token_logits = torch.cat(seg_token_logits, dim=0) # [24, 18, 27]
    seg_loc_logits = torch.cat(seg_loc_logits, dim=0) # [24, 18, 27]
    output_token_logits = torch.cat(output_token_logits, dim=0) # [24, 18, 27]
    video = torch.cat(video, dim=0)  # [24, 3, 720, 1080]
    
    #########################################################################
    # 1. 将seg_token_logits， seg_loc_logits，output_token_logits分别转化为热力图
    # 2. 将热力图分别与video叠加，可视化出来
    #########################################################################
    def visualize_logits_overlay(
        logits_tensor: torch.Tensor,
        video_tensor: torch.Tensor,
        clip_len: int,
        title: str,
        save_path: str,
        alpha: float = 0.6,
        colormap: int = cv2.COLORMAP_JET,
    ):
        """
        一个辅助函数，用于将logits可视化为热力图并叠加到视频帧上。
    
        Args:
            logits_tensor (torch.Tensor): 形状为 [N, S, V] 的Logits张量, N是片段数。
            video_tensor (torch.Tensor): 形状为 [T, C, H, W] 的视频张量, T是总帧数。
            clip_len (int): 每个视频片段包含的帧数。
            title (str): 可视化图像的标题。
            save_path (str): 图像保存路径。
            alpha (float): 热力图的透明度。
            colormap (int): 要使用的OpenCV颜色映射。
        """
        # 步骤1: 准备用于OpenCV处理的视频帧
        video_np = video_tensor.cpu().permute(0, 2, 3, 1).numpy() # T, H, W, C
        if video_np.max() <= 1.0:
            video_np = (video_np * 255).astype(np.uint8)
        else:
            video_np = np.clip(video_np, 0, 255).astype(np.uint8)
    
        # 步骤2: 逐片段处理Logits并生成叠加后的帧
        num_clips = logits_tensor.shape[0]
        total_frames, H, W, _ = video_np.shape
        overlaid_frames = []
    
        for i in range(num_clips):
            # a. 提取当前片段的logits并归一化
            logit_clip = logits_tensor[i].float().cpu().numpy()
            min_val, max_val = logit_clip.min(), logit_clip.max()
            if max_val > min_val:
                logit_normalized = (255 * (logit_clip - min_val) / (max_val - min_val)).astype(np.uint8)
            else:
                logit_normalized = np.zeros_like(logit_clip, dtype=np.uint8)
    
            # b. 应用颜色映射生成热力图，并调整大小以匹配视频帧
            heatmap = cv2.applyColorMap(logit_normalized, colormap)
            heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
    
            # c. 将此热力图应用到该片段对应的所有帧上
            start_idx = i * clip_len
            end_idx = min((i + 1) * clip_len, total_frames)
            for j in range(start_idx, end_idx):
                original_frame = video_np[j]
                # 使用cv2.addWeighted进行透明叠加
                overlaid_frame = cv2.addWeighted(original_frame, 1 - alpha, heatmap_resized, alpha, 0)
                overlaid_frames.append(cv2.cvtColor(overlaid_frame, cv2.COLOR_BGR2RGB)) # 转为RGB以便matplotlib显示
    
        # 步骤3: 使用Matplotlib创建并保存可视化结果
        # 为了避免图像过大，我们选择性地展示一些帧
        num_frames_to_show = min(len(overlaid_frames), 8) # 最多显示8帧
        indices = np.linspace(0, len(overlaid_frames) - 1, num_frames_to_show, dtype=int)
        
        fig, axes = plt.subplots(2, num_frames_to_show, figsize=(num_frames_to_show * 3, 6.5))
        fig.suptitle(title, fontsize=18)
    
        for i, idx in enumerate(indices):
            # 显示原始帧
            axes[0, i].imshow(cv2.cvtColor(video_np[idx], cv2.COLOR_BGR2RGB))
            axes[0, i].set_title(f'Frame {idx} (Original)')
            axes[0, i].axis('off')
    
            # 显示叠加热力图的帧
            axes[1, i].imshow(overlaid_frames[idx])
            axes[1, i].set_title(f'Frame {idx} (Overlay)')
            axes[1, i].axis('off')
    
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Logits热力图已保存至: {save_path}")

    # 分别为三种logits调用上面的可视化函数
    visualize_logits_overlay(
        logits_tensor=seg_token_logits,
        video_tensor=video,
        clip_len=clip_len,
        title="Heatmap Overlay for Seg Token Logits",
        save_path="./heatmap_seg_token_logits.png"
    )
    
    visualize_logits_overlay(
        logits_tensor=seg_loc_logits,
        video_tensor=video,
        clip_len=clip_len,
        title="Heatmap Overlay for Seg Loc Logits",
        save_path="./heatmap_seg_loc_logits.png"
    )
    
    visualize_logits_overlay(
        logits_tensor=output_token_logits,
        video_tensor=video,
        clip_len=clip_len,
        title="Heatmap Overlay for Output Token Logits",
        save_path="./heatmap_output_token_logits.png"
    )
    # target_labels = (anomaly_masks == 1).any(dim=(1, 2)).int()[:len(pred_scores)]
    print(torch.max(pred_masks))
    visualize_video_with_mask(video_tensor=video[::2], mask_tensor=pred_masks[::2], save_path="./demo.png")
    plot_pred_score_with_windows(pred_scores, anomaly_windows)
    # visualize_predictions(ori_video, normal_pred_masks, anomaly_pred_masks, normal_masks, anomaly_masks, pred_score, 'demo.png')
    
if __name__ == '__main__':
    
    demo(model=None, tokenizer=None, processor=None)
