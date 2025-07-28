from typing import Dict
import numba
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
import pdb

class RobotImageDataset(BaseImageDataset):
    def __init__(self,
                 zarr_path, 
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 batch_size=64,
                 max_train_episodes=None):
        """
        初始化数据集对象，配置采样器，数据缓冲区等。

        :param zarr_path: 数据集的路径，包含回放缓存（replay buffer）的文件。
        :param horizon: 采样序列的长度。
        :param pad_before: 序列前面需要填充的空白。
        :param pad_after: 序列后面需要填充的空白。
        :param seed: 随机种子，用于生成验证集掩码。
        :param val_ratio: 验证集所占比例。
        :param batch_size: 批次大小。
        :param max_train_episodes: 最大训练集的集数，如果为 `None` 则没有限制。
        """
        
        # 加载回放缓存，指定数据键（包括头部摄像头图像、状态和动作）
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=['head_camera', 'state', 'action']  # 选择需要的字段
        )
            
        # 生成验证集掩码
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask  # 训练集掩码（对验证集掩码取反）
        
        # 根据最大训练集数量对训练集掩码进行下采样
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        # 创建 SequenceSampler，控制采样序列的长度和填充方式
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        # 保存训练集掩码、序列长度和填充参数
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.batch_size = batch_size

        # 初始化缓冲区，存储每个样本的序列
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        # 将缓冲区从 NumPy 转换为 PyTorch 张量，并固定内存
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

    def get_validation_dataset(self):
        """
        获取验证数据集对象，通过复制当前数据集并使用验证集掩码创建新的采样器。

        :return: 验证数据集对象
        """
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask  # 使用反转的训练集掩码
        )
        val_set.train_mask = ~self.train_mask  # 验证集掩码
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        """
        获取数据归一化器，返回用于归一化动作和状态的归一化器。

        :param mode: 归一化模式（如 'limits'）。
        :param kwargs: 其他归一化器的参数。
        :return: 正常化器对象
        """
        # 从回放缓存中提取数据
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state']
        }
        # 创建并拟合归一化器
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # 为不同的相机图像获取归一化器
        normalizer['head_cam'] = get_image_range_normalizer()
        normalizer['front_cam'] = get_image_range_normalizer()
        normalizer['left_cam'] = get_image_range_normalizer()
        normalizer['right_cam'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        """
        返回数据集的大小（即采样器的大小）。

        :return: 数据集大小
        """
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """
        将样本从原始数据格式转换为标准化的字典格式。

        :param sample: 一个样本，包含摄像头图像、状态和动作。
        :return: 标准化后的数据字典
        """
        agent_pos = sample['state'].astype(np.float32)  # 处理状态信息
        head_cam = np.moveaxis(sample['head_camera'], -1, 1) / 255  # 归一化头部摄像头图像
        # 处理其他相机图像（如果需要）
        # front_cam = np.moveaxis(sample['front_camera'], -1, 1) / 255
        # left_cam = np.moveaxis(sample['left_camera'], -1, 1) / 255
        # right_cam = np.moveaxis(sample['right_camera'], -1, 1) / 255

        # 将处理后的数据存储在字典中
        data = {
            'obs': {
                'head_cam': head_cam,  # T, 3, H, W
                # 'front_cam': front_cam, # T, 3, H, W
                # 'left_cam': left_cam, # T, 3, H, W
                # 'right_cam': right_cam, # T, 3, H, W
                'agent_pos': agent_pos,  # T, D
            },
            'action': sample['action'].astype(np.float32)  # T, D
        }
        return data

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        通过索引获取数据集中的一个样本，并转换为 PyTorch 张量格式。

        :param idx: 索引，支持整数、切片和批量索引。
        :return: 转换后的数据字典
        """
        # import pdb; pdb.set_trace()
        if isinstance(idx, slice):
            raise NotImplementedError  # 尚未实现切片索引
        elif isinstance(idx, int):
            # 获取单个样本并转换为 PyTorch 张量
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            # 批量索引
            assert len(idx) == self.batch_size
            # 从回放缓存中批量采样数据
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(self.buffers[k], v, self.sampler.indices, idx, self.sampler.sequence_length)
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        """
        将采样的数据转换到设备并标准化。

        :param samples: 采样的数据。
        :param device: PyTorch 设备。
        :return: 经过标准化后的数据字典
        """
        agent_pos = samples['state'].to(device, non_blocking=True)
        head_cam = samples['head_camera'].to(device, non_blocking=True) / 255.0
        # front_cam = samples['front_camera'].to(device, non_blocking=True) / 255.0
        # left_cam = samples['left_camera'].to(device, non_blocking=True) / 255.0
        # right_cam = samples['right_camera'].to(device, non_blocking=True) / 255.0
        action = samples['action'].to(device, non_blocking=True)
        
        return {
            'obs': {
                'head_cam': head_cam,  # B, T, 3, H, W
                # 'front_cam': front_cam, # B, T, 3, H, W
                # 'left_cam': left_cam, # B, T, 3, H, W
                # 'right_cam': right_cam, # B, T, 3, H, W
                'agent_pos': agent_pos,  # B, T, D
            },
            'action': action  # B, T, D
        }

def _batch_sample_sequence(data: np.ndarray, input_arr: np.ndarray, indices: np.ndarray, idx: np.ndarray, sequence_length: int):
    """
    批量采样序列，将数据从回放缓存复制到目标数据缓冲区。

    :param data: 目标数据数组。
    :param input_arr: 输入数组（来自回放缓存）。
    :param indices: 索引数组，用于确定每个序列的起始和结束位置。
    :param idx: 需要采样的索引。
    :param sequence_length: 序列长度。
    """
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        # 将数据从缓存复制到目标数据数组
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]  # 前部分填充
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]  # 后部分填充

# 并行和顺序版本的采样函数
_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)

def batch_sample_sequence(data: np.ndarray, input_arr: np.ndarray, indices: np.ndarray, idx: np.ndarray, sequence_length: int):
    """
    根据批次大小选择适当的并行或顺序版本的批量采样。

    :param data: 目标数据数组。
    :param input_arr: 输入数据数组。
    :param indices: 索引数组。
    :param idx: 当前批次的索引。
    :param sequence_length: 序列长度。
    """
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2 ** 16:
        # 如果数据较大，使用并行版本
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        # 否则使用顺序版本
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)
