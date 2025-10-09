# models/cnn_rnn_ae.py
import torch
import torch.nn as nn

class CNNRNNEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim, cnn_channels=32, rnn_hidden=64, rnn_layers=1):
        super().__init__()
        self.cnn = nn.Conv1d(in_channels, cnn_channels, kernel_size=3, padding=1)
        self.rnn = nn.GRU(cnn_channels, rnn_hidden, num_layers=rnn_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(rnn_hidden*2, latent_dim)

    def forward(self, x):  # x: (B, T, C)
        x = x.permute(0,2,1)  # (B, C, T)
        x = self.cnn(x)       # (B, cnn_channels, T)
        x = x.permute(0,2,1)  # (B, T, cnn_channels)
        out, _ = self.rnn(x)  # (B, T, 2*rnn_hidden)
        pooled = out.mean(dim=1)
        return self.fc(pooled)

class RNNDecoder(nn.Module):
    def __init__(self, out_channels, latent_dim, rnn_hidden=64, rnn_layers=1, T=100):
        super().__init__()
        self.T = T
        self.fc = nn.Linear(latent_dim, rnn_hidden*2)
        self.rnn = nn.GRU(rnn_hidden*2, rnn_hidden, num_layers=rnn_layers, batch_first=True)
        self.head = nn.Linear(rnn_hidden, out_channels)
    def forward(self, z):
        # expand z to sequence
        seq = self.fc(z).unsqueeze(1).repeat(1, self.T, 1)
        out, _ = self.rnn(seq)
        return self.head(out)  # (B, T, out_channels)

class CNNRNNAutoEncoder(nn.Module):
    def __init__(self, T, C, latent_dim=32):
        super().__init__()
        self.encoder = CNNRNNEncoder(in_channels=C, latent_dim=latent_dim)
        self.decoder = RNNDecoder(out_channels=C, latent_dim=latent_dim, T=T)

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z
