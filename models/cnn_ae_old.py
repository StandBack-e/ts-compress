# models/cnn_ae.py
import torch
import torch.nn as nn

class Encoder1DCNN(nn.Module):
    def __init__(self, in_channels, latent_dim, base_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1, stride=2),  # downsample
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels*2, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # global aggregation -> (B, C2, 1)
            nn.Flatten(),
            nn.Linear(base_channels*2, latent_dim)
        )

    def forward(self, x):  # x: (B, T, C)
        x = x.permute(0,2,1)  # -> (B, C, T)
        return self.net(x)    # -> (B, latent_dim)

class Decoder1DCNN(nn.Module):
    def __init__(self, out_channels, latent_dim, base_channels=32, out_len=100):
        super().__init__()
        self.fc = nn.Linear(latent_dim, base_channels*4)  # 改成更大的 channels
        self.out_len = out_len
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(base_channels*4, base_channels*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(base_channels*2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(base_channels, out_channels, kernel_size=4, stride=2, padding=1),
            # no activation
        )

    def forward(self, z):
        x = self.fc(z).unsqueeze(-1)  # (B, C, 1)
        x = self.deconv(x)            # -> (B, out_channels, T_est)
        x = x[:, :, :self.out_len]    # crop/pad
        return x.permute(0,2,1)      # -> (B, T, C)


class CNNAutoEncoder(nn.Module):
    def __init__(self, T, C, latent_dim=16, base_channels=32):
        super().__init__()
        self.encoder = Encoder1DCNN(in_channels=C, latent_dim=latent_dim, base_channels=base_channels)
        self.decoder = Decoder1DCNN(out_channels=C, latent_dim=latent_dim, base_channels=base_channels, out_len=T)

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z
