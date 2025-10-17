# models/patchtst_ae.py
import torch
import torch.nn as nn

class Patching(nn.Module):
    """ 将时间序列窗口切分成块 (Patching) """
    def __init__(self, patch_len, stride):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x):
        # x: [Batch, Time, Channels]
        n_patches = (x.shape[1] - self.patch_len) // self.stride + 1
        # [Batch, n_patches, patch_len]
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        return patches

class PatchTSTAutoEncoder(nn.Module):
    def __init__(self, T, C, latent_dim, patch_len=16, patch_stride=8, 
                 d_model=128, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=256):
        super().__init__()
        
        self.T = T
        self.C = C
        self.patch_len = patch_len
        
        # 1. Patching
        # 确保 C=1, 因为PatchTST通常用于单变量或通道独立的场景
        assert C == 1, "PatchTSTAutoEncoder currently only supports single-channel (C=1) time series."
        self.patching = Patching(patch_len, patch_stride)
        
        # 计算Patch数量
        self.n_patches = (T - patch_len) // patch_stride + 1
        
        # 2. Input Projection
        # 将每个Patch (长度为patch_len) 映射到 d_model 维度
        self.input_projection = nn.Linear(patch_len, d_model)
        
        # 3. Positional Encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, self.n_patches, d_model))
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # 5. Latent Projection
        # 将encoder的输出 (n_patches * d_model) 压平成一个向量，然后映射到latent_dim
        self.flatten = nn.Flatten(start_dim=1)
        self.to_latent = nn.Linear(self.n_patches * d_model, latent_dim)

        # --- 解码器部分 ---
        self.from_latent = nn.Linear(latent_dim, self.n_patches * d_model)
        
        # 6. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # 7. Output Projection
        self.output_projection = nn.Linear(d_model, patch_len)

    def encode(self, x):
        # x: [B, T, C]
        x = x.squeeze(-1) # -> [B, T]
        
        # 1. Patching
        patches = self.patching(x) # -> [B, n_patches, patch_len]
        
        # 2. Projection and Positional Encoding
        x = self.input_projection(patches) # -> [B, n_patches, d_model]
        x = x + self.pos_encoder
        
        # 3. Transformer Encoder
        memory = self.transformer_encoder(x) # -> [B, n_patches, d_model]
        
        # 4. To Latent
        z = self.flatten(memory)
        z = self.to_latent(z) # -> [B, latent_dim]
        return z

    def decode(self, z):
        # z: [B, latent_dim]
        
        # 1. From Latent
        x = self.from_latent(z)
        x = x.view(-1, self.n_patches, self.input_projection.out_features) # -> [B, n_patches, d_model]

        # 2. Transformer Decoder
        # 创建一个与memory形状相同的目标序列(tgt)，通常用0初始化
        tgt = torch.zeros_like(x)
        output = self.transformer_decoder(tgt, x) # -> [B, n_patches, d_model]
        
        # 3. Output Projection to Patches
        rec_patches = self.output_projection(output) # -> [B, n_patches, patch_len]
        
        # 4. Unpatching (简单拼接)
        # 这是一个简化的unpatching，更复杂的方法会处理重叠
        rec_seq = rec_patches.flatten(start_dim=1) # -> [B, n_patches * patch_len]
        
        # 裁剪到原始长度T
        rec_seq = rec_seq[:, :self.T].unsqueeze(-1) # -> [B, T, 1]
        
        return rec_seq

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z