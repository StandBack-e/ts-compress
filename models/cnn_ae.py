# models/cnn_ae.py
import torch
import torch.nn as nn

class Encoder1DCNN(nn.Module):
    def __init__(self, in_channels, latent_dim, base_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1, stride=2),  # T -> T/2
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels*2, kernel_size=3, padding=1, stride=2), # T/2 -> T/4
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # global agg -> (B, C2, 1)
            nn.Flatten(),             # -> (B, C2)
            nn.Linear(base_channels*2, latent_dim)
        )

    def forward(self, x):  # x: (B, T, C)
        x = x.permute(0,2,1)  # -> (B, C, T)
        return self.net(x)    # -> (B, latent_dim)

class DecoderConvTranspose(nn.Module):
    def __init__(self, out_channels, latent_dim, base_channels=32, out_len=32):
        super().__init__()
        # we assume encoder downsamples by factor 4 (two stride=2), so initial low-res length = out_len // 4
        if out_len % 4 != 0:
            raise ValueError("out_len must be divisible by 4 for this decoder (choose T divisible by 4).")
        self.low_res_len = out_len // 4
        self.base_c2 = base_channels * 2
        # fc now expands latent -> (B, base_c2 * low_res_len)
        self.fc = nn.Linear(latent_dim, self.base_c2 * self.low_res_len)
        self.deconv = nn.Sequential(
            # input (B, base_c2, low_res_len)
            nn.ConvTranspose1d(self.base_c2, base_channels, kernel_size=4, stride=2, padding=1), # *2
            nn.ReLU(),
            nn.ConvTranspose1d(base_channels, out_channels, kernel_size=4, stride=2, padding=1),  # *2 -> out_len
            # no activation (regression)
        )
        self.out_len = out_len

    def forward(self, z):
        B = z.size(0)
        x = self.fc(z)                       # (B, base_c2 * low_res_len)
        x = x.view(B, self.base_c2, self.low_res_len)  # (B, base_c2, low_res_len)
        x = self.deconv(x)                   # (B, out_channels, out_len)
        x = x[:, :, :self.out_len]           # safety crop if needed
        return x.permute(0,2,1)              # -> (B, T, C)

class CNNAutoEncoder(nn.Module):
    def __init__(self, T, C, latent_dim=16, base_channels=32):
        super().__init__()
        self.encoder = Encoder1DCNN(in_channels=C, latent_dim=latent_dim, base_channels=base_channels)
        self.decoder = DecoderConvTranspose(out_channels=C, latent_dim=latent_dim, base_channels=base_channels, out_len=T)

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z
