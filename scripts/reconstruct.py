# scripts/reconstruct.py
import numpy as np
import json, os

def reconstruct_from_windows(windows, window, stride):
    """
    windows: np.array (N_windows, T, C)  或 (N_windows, T)
    window: int, T
    stride: int, S

    returns:
      seq: np.array (L, C)  or (L,)
      coverage: np.array (L,)  每个时间点被覆盖的次数（用于 debug）
    Notes:
      L = (N_windows - 1) * stride + window
    """
    if windows.ndim == 2:
        windows = windows[:, :, None]  # to (N, T, 1)

    N, T, C = windows.shape
    assert T == window, f"window mismatch T={T} vs window arg {window}"
    L = (N - 1) * stride + window
    seq = np.zeros((L, C), dtype=windows.dtype)
    coverage = np.zeros((L,), dtype=np.int32)

    for i in range(N):
        start = i * stride
        end = start + window
        seq[start:end] += windows[i]
        coverage[start:end] += 1

    # avoid division by zero (shouldn't happen)
    mask = coverage > 0
    seq[mask] = seq[mask] / coverage[mask, None]
    return seq.squeeze(), coverage

def inverse_normalize(seq, norm_stats_path):
    """
    seq: (L, C) 或 (L,) 已经是 numpy
    norm_stats_path: path to json with {'mean':..., 'scale':...}
    """
    if not os.path.exists(norm_stats_path):
        raise FileNotFoundError(norm_stats_path)
    stats = json.load(open(norm_stats_path))
    mean = float(stats['mean'])
    scale = float(stats['scale'])
    return seq * scale + mean

def add_linear_trend(seq, slope, intercept):
    # seq shape (L,) or (L,C)
    L = seq.shape[0]
    t = np.arange(L)
    trend = slope * t + intercept
    if seq.ndim == 2:
        return seq + trend[:, None]
    return seq + trend

def estimate_and_add_trend(seq):
    L = seq.shape[0]
    t = np.arange(L)
    if seq.ndim == 2:
        # 对每个 channel 单独拟合并加回
        for c in range(seq.shape[1]):
            y = seq[:, c]
            A = np.vstack([t, np.ones(L)]).T
            m, b = np.linalg.lstsq(A, y, rcond=None)[0]
            seq[:, c] = seq[:, c] + (m * t + b)
        return seq
    else:
        y = seq
        A = np.vstack([t, np.ones(L)]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return seq + (m * t + b)


