# models/controller.py
import torch
import torch.nn as nn

class ComplexityEstimator(nn.Module):
    """
    从原始分片或浅特征估计最佳 latent_dim 比例（continuous），训练目标为重构误差与压缩率加权
    这里只是一个轻量 MLP：输入可为统计特征（variance, entropy, kurtosis）
    """
    def __init__(self, in_dim=10, hidden=64, out_scale=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()  # 输出 0..1, 用来 scale latent_dim
        )
        self.out_scale = out_scale

    def forward(self, feats):
        # feats: (B, in_dim)
        ratio = self.net(feats).squeeze(-1)
        return ratio * self.out_scale
