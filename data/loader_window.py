# data/loaders.py (修改后)
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, arr):
        self.windows = torch.from_numpy(arr).float()

    def __len__(self):
        # 因为我们每次需要获取(i, i+1)两个窗口，所以总长度要减1
        return self.windows.shape[0] - 1

    def __getitem__(self, idx):
        # 返回一对连续的窗口
        # window1 是当前窗口
        # window2 是紧接着的下一个窗口
        window1 = self.windows[idx]
        window2 = self.windows[idx+1]
        return window1, window2

def get_dataloaders(train_arr, val_arr, batch_size):
    train_ds = TimeSeriesDataset(train_arr)
    val_ds = TimeSeriesDataset(val_arr)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader