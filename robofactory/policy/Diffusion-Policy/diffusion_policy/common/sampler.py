from typing import Optional
import numpy as np
import numba
from diffusion_policy.common.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
def create_indices(
    episode_ends: np.ndarray, sequence_length: int, 
    episode_mask: np.ndarray,
    pad_before: int = 0, pad_after: int = 0,
    debug: bool = True) -> np.ndarray:
    """
    创建采样序列的索引。该函数为每个 episode 创建连续的采样窗口（即一个个子序列） 
    对应回放缓冲区中的数据。

    :param episode_ends: 一个包含每个 episode 结束位置的数组
    :param sequence_length: 每个采样序列的长度
    :param episode_mask: 一个布尔数组，标记哪些 episode 被用于训练（True 表示该 episode 被使用）
    :param pad_before: 每个序列前的填充长度（默认为 0）
    :param pad_after: 每个序列后的填充长度（默认为 0）
    :param debug: 是否启用调试检查（默认为 True）
    
    :return: 返回一个包含索引的数组，每个索引包含四个值：
             [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
             这四个值分别表示从缓冲区中选取的子序列的开始和结束索引，以及在该子序列中的有效部分的起始和结束索引。
    """
    
    episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # 如果该 episode 不被使用，跳过
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # 循环生成所有可能的子序列索引
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert(start_offset >= 0)  # 调试检查
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    """
    根据指定的验证集比例生成验证集掩码。

    :param n_episodes: 训练集中的 episode 数量
    :param val_ratio: 验证集所占比例（0 到 1 之间）
    :param seed: 随机种子（默认为 0）
    
    :return: 返回一个布尔数组，表示哪些 episode 用于验证集（True 表示验证集）。
    """
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # 至少包含 1 个验证集 episode 和 1 个训练集 episode
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    # 随机选择 n_val 个验证集 episode
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    """
    对训练数据进行下采样，保证训练集数量不超过 max_n。

    :param mask: 训练集的布尔掩码，表示哪些 episode 用于训练
    :param max_n: 最大训练集大小
    :param seed: 随机种子（默认为 0）
    
    :return: 返回一个新的训练集掩码，保证其大小不超过 max_n。
    """
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train  # 确保下采样后的训练集大小正确
    return train_mask


class SequenceSampler:
    def __init__(self, 
        replay_buffer: ReplayBuffer, 
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray] = None,
        ):
        """
        序列采样器，用于从回放缓冲区中采样固定长度的序列。

        :param replay_buffer: 回放缓冲区，包含训练数据。
        :param sequence_length: 采样序列的长度。
        :param pad_before: 序列前的填充长度（默认为 0）。
        :param pad_after: 序列后的填充长度（默认为 0）。
        :param keys: 要使用的键（例如图像、状态、动作等），如果为 None，则使用所有键。
        :param key_first_k: 字典，指定对于每个键（key）只使用前 k 个数据来提高性能。
        :param episode_mask: 布尔数组，表示哪些 episode 被用于训练（如果为 None，则默认所有 episode 都被使用）。
        """
        super().__init__()
        assert(sequence_length >= 1)  # 确保序列长度大于等于 1
        if keys is None:
            keys = list(replay_buffer.keys())
        
        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(episode_ends, 
                sequence_length=sequence_length, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
                )
        else:
            indices = np.zeros((0, 4), dtype=np.int64)

        # 采样的索引（缓冲区的开始和结束索引，以及样本的开始和结束索引）
        self.indices = indices 
        self.keys = list(keys)  # 防止 OmegaConf 在使用列表时出现性能问题
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
    
    def __len__(self):
        """
        返回采样器中索引的数量，即可采样的序列数量。
        """
        return len(self.indices)
        
    def sample_sequence(self, idx):
        """
        根据索引从回放缓冲区采样一个序列。

        :param idx: 需要采样的序列索引。
        :return: 返回一个字典，包含每个键（如状态、图像等）的序列数据。
        """

        
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # 性能优化，避免不必要的小数据分配
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # 只加载前 k 个数据，避免不必要的内存分配
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                sample = np.full((n_data,) + input_arr.shape[1:], 
                    fill_value=np.nan, dtype=input_arr.dtype)
                try:
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
                except Exception as e:
                    import pdb; pdb.set_trace()
            data = sample
            # 对采样序列进行填充
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result
