# utils.py
import numpy as np
import torch
import torch.nn.functional as F

def weighted_mse(x, y, weight=None):
    # x, y: (B, T, C)
    diff = (x - y) ** 2
    if weight is None:
        return diff.mean()
    return (diff * weight).mean()

def extract_shallow_features(x):
    # x: (B, T, C) numpy or tensor. 返回 (B, in_dim)
    if isinstance(x, np.ndarray):
        arr = x
    else:
        arr = x.detach().cpu().numpy()
    B, T, C = arr.shape
    feats = []
    for b in range(B):
        sample = arr[b]
        var = sample.var(axis=0).mean()
        mean = sample.mean(axis=0).mean()
        maxv = sample.max(axis=0).mean()
        minv = sample.min(axis=0).mean()
        energy = (sample**2).mean()
        # 简单统计
        feats.append([var, mean, maxv, minv, energy])
    return np.array(feats, dtype=np.float32)
