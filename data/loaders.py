# data/loaders.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    """
    假设数据已预处理成 numpy arrays: shape (N, T, C)
    N: samples, T: time steps per slice (e.g., 5s -> depends on freq), C: channels/sensors
    """
    def __init__(self, arr):
        assert isinstance(arr, np.ndarray)
        self.x = arr.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]  # (T, C)

def get_dataloaders(train_arr, val_arr, batch_size=64, num_workers=4):
    train_ds = TimeSeriesDataset(train_arr)
    val_ds = TimeSeriesDataset(val_arr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
