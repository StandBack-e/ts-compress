# models/cnn_rnn_attention_ae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. 向量量化核心层 (Vector Quantizer Layer)
# 这是实现VQ-VAE的基础模块。
# ==============================================================================
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # 初始化码本(codebook)，这是一个可学习的参数
        self.embedding = nn.Embedding(self.codebook_size, self.embedding_dim)
        # 初始化权重为均匀分布
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z):
        # z: 从编码器输出的连续潜向量, shape: [B, D]
        
        # 调整z的形状以计算距离: [B, D] -> [B, 1, D]
        z_reshaped = z.unsqueeze(1)
        
        # 码本权重, shape: [K, D]
        embedding_weight = self.embedding.weight
        
        # 计算z与码本中每个码字的L2距离
        # (a-b)^2 = a^2 + b^2 - 2ab
        distances = (torch.sum(z_reshaped**2, dim=2, keepdim=True) 
                     + torch.sum(embedding_weight**2, dim=1)
                     - 2 * torch.matmul(z_reshaped, embedding_weight.t()))
        
        # 找到距离最近的码字的索引
        # shape: [B, 1]
        encoding_indices = torch.argmin(distances, dim=2)
        
        # 将索引转换为one-hot编码，以便从码本中取出对应的码字
        # shape: [B, K]
        indices_squeezed = encoding_indices.squeeze(1)
        encoding_one_hot = F.one_hot(indices_squeezed, self.codebook_size).type(z.dtype)
        
        # 取出量化后的向量(最近的码字)
        # shape: [B, D]
        z_quantized = torch.matmul(encoding_one_hot, self.embedding.weight)
        
        # --- 计算损失 ---
        # 核心技巧：使用直通估计器(Straight-Through Estimator, STE)
        # 在前向传播中使用z_quantized，但在反向传播时，梯度会直接从z_quantized“跳”到z上
        # 这样编码器就能接收到来自解码器的梯度
        z_quantized_ste = z + (z_quantized - z).detach()

        # 1. Commitment Loss: 鼓励编码器的输出z更接近它所选择的码字
        commitment_loss = F.mse_loss(z.detach(), z_quantized)
        
        # 2. Codebook Loss (字典学习损失): 通过来自解码器的梯度来更新码本
        codebook_loss = F.mse_loss(z, z_quantized.detach())
        
        # 组合损失
        loss = codebook_loss + self.commitment_cost * commitment_loss
        
        return z_quantized_ste, encoding_indices, loss

# ==============================================================================
# 2. 残差量化模块 (Residual VQ)
# ==============================================================================
class ResidualVQ(nn.Module):
    def __init__(self, num_quantizers, codebook_size, latent_dim, commitment_cost=0.25):
        super().__init__()
        self.num_quantizers = num_quantizers
        
        # 创建一个包含多个独立VQ层的列表
        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, latent_dim, commitment_cost) 
            for _ in range(self.num_quantizers)
        ])

    def forward(self, z):
        # z: 原始潜向量
        residual = z
        all_quantized_ste = []
        all_indices = []
        total_loss = 0

        for quantizer in self.quantizers:
            # 对当前残差进行量化
            quantized_ste, indices, loss = quantizer(residual)
            
            # 使用STE版本的quantized向量来更新残差，以保证梯度可以回传
            # 注意：这里更新残差用的是不带梯度的z_quantized
            # quantized = self.quantizers[0].embedding(indices) # 直接从码本取，避免梯度问题
            # residual = residual - quantized

            # **关键修正**: 更新下一级的残差
            # 我们减去的是带有梯度的量化向量的 detach() 版本
            # 这样可以保证梯度流的正确性和训练的稳定性
            residual = residual - quantized_ste.detach()
            
            all_quantized_ste.append(quantized_ste)
            all_indices.append(indices)
            total_loss += loss

        # 最终重构的z是所有量化向量(STE版本)的和
        z_reconstructed = torch.stack(all_quantized_ste, dim=0).sum(dim=0)
        
        # 最终的码字索引是多个索引的组合
        final_indices = torch.stack(all_indices, dim=1) # shape: [B, num_quantizers]
        
        return z_reconstructed, final_indices, total_loss
    
class CNNRNNAttentionAutoEncoder(nn.Module):
    """
    一个结合了CNN、RNN和Attention的混合自编码器模型。
    - CNN部分用于从时间序列中提取局部的、空间上的特征。
    - RNN部分用于学习这些特征在时间上的依赖关系。
    """
    def __init__(self, T, C, latent_dim=16, base_channels=32, rnn_hidden_dim=64, nhead=4,
                 use_vq=False, num_quantizers=4, codebook_size=256):
        """
        初始化函数
        参数:
            T (int): 输入时间序列的长度 (time steps)
            C (int): 输入时间序列的特征维度 (channels)
            latent_dim (int): 潜在空间的维度
            base_channels (int): CNN第一层的通道数
            rnn_hidden_dim (int): RNN隐藏层的维度
        """
        super().__init__()
        self.T = T
        self.C = C
        self.use_vq = use_vq # 是否启用量化
        
        # ------------------- 编码器 (Encoder) -------------------
        # 1. CNN特征提取器
        self.cnn_encoder = nn.Sequential(
            # 输入: (B, C, T)
            nn.Conv1d(C, base_channels, kernel_size=3, padding=1, stride=2),  # T -> T/2
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels*2, kernel_size=3, padding=1, stride=2), # T/2 -> T/4
            nn.ReLU(),
            # 输出: (B, base_channels*2, T/4)
        )
        
        # 计算CNN输出的序列长度和通道数
        # cnn_out_seq_len = T // 4
        cnn_out_channels = base_channels * 2


        # --- 新增：Transformer Encoder ---
        # 负责对CNN提取的局部特征序列进行全局关系建模
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=cnn_out_channels, 
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.attention_encoder = nn.TransformerEncoder(transformer_layer, num_layers=2) # 可尝试1-3层
        
        # 2. RNN时序编码器
        # 使用双向GRU来捕捉更丰富的上下文信息
        self.rnn_encoder = nn.GRU(
            input_size=cnn_out_channels, 
            hidden_size=rnn_hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True
        )
        
        # 3. 将RNN的最终隐藏状态映射到潜在空间
        # 因为是双向GRU，所以维度是 rnn_hidden_dim * 2
        self.fc_latent = nn.Linear(rnn_hidden_dim * 2, latent_dim)

        # =========== 新增：RQ模块 ===========
        if self.use_vq:
            self.residual_vq = ResidualVQ(
                num_quantizers=num_quantizers,
                codebook_size=codebook_size,
                latent_dim=latent_dim
            )
        # ============================================

         # =========== 新增：潜在空间预测器 ============
        # 一个简单的线性层，输入和输出维度都是 latent_dim
        self.latent_predictor = nn.Linear(latent_dim, latent_dim)
        # ============================================

        # ------------------- 解码器 (Decoder) -------------------
        # 1. 将潜在向量映射回RNN隐藏状态的维度
        self.fc_decoder_in = nn.Linear(latent_dim, rnn_hidden_dim)

        # 2. RNN时序生成器
        self.rnn_decoder = nn.GRU(
            input_size=rnn_hidden_dim, 
            hidden_size=rnn_hidden_dim, 
            num_layers=2, 
            batch_first=True
        )
        
        # 3. 反卷积层 (ConvTranspose) 用于上采样
        self.deconv_decoder = nn.Sequential(
            # 输入: (B, rnn_hidden_dim, T)
            nn.ConvTranspose1d(rnn_hidden_dim, base_channels, kernel_size=4, stride=2, padding=1), # T -> T*2
            nn.ReLU(),
            nn.ConvTranspose1d(base_channels, C, kernel_size=4, stride=2, padding=1), # T*2 -> T*4
            # 输出: (B, C, T)
        )

    def encode(self, x):
        """
        前向传播函数
        输入 x: (B, T, C)
        """
        # --- 编码过程 ---
        x_permuted = x.permute(0, 2, 1)  # -> (B, C, T)
        
        # 1. 通过CNN提取特征序列
        cnn_out = self.cnn_encoder(x_permuted)  # -> (B, cnn_out_channels, T/4)
        cnn_out = cnn_out.permute(0, 2, 1)     # -> (B, T/4, cnn_out_channels)


        # --- 核心改动 ---
        # 将特征序列送入Transformer进行处理
        attn_out = self.attention_encoder(cnn_out)

        # 将经过注意力加权的特征送入RNN
        _, h_n = self.rnn_encoder(attn_out)
        
        # 2. 通过RNN编码时序信息
        # h_n 的形状: (num_layers * num_directions, B, rnn_hidden_dim)
        # _, h_n = self.rnn_encoder(cnn_out)
        
        # 3. 组合双向RNN的最终隐藏状态
        # h_n可以被重塑为 (num_layers, num_directions, B, rnn_hidden_dim)
        # 我们取最后一层的两个方向的隐藏状态
        h_n_last_layer = h_n.view(2, 2, x.size(0), -1)[-1] # (num_directions, B, rnn_hidden_dim)
        h_n_forward = h_n_last_layer[0]  # 前向
        h_n_backward = h_n_last_layer[1] # 后向
        z_combined = torch.cat([h_n_forward, h_n_backward], dim=1)  # -> (B, rnn_hidden_dim * 2)
        
        # 4. 得到最终的潜在向量 z
        z = self.fc_latent(z_combined)  # -> (B, latent_dim)
        return z
    
    def decode(self, z):
        """
        新增的解码函数。
        输入 z: (B, latent_dim)
        输出 x_rec: (B, T, C)
        """
        decoder_h0 = self.fc_decoder_in(z)
        decoder_h0 = decoder_h0.unsqueeze(0).repeat(2, 1, 1)

        decoder_input_len = self.T // 4
        decoder_input = torch.zeros(z.size(0), decoder_input_len, self.rnn_decoder.input_size, device=z.device)

        rnn_out, _ = self.rnn_decoder(decoder_input, decoder_h0)
        
        rnn_out = rnn_out.permute(0, 2, 1)
        x_rec = self.deconv_decoder(rnn_out)
        
        x_rec = x_rec.permute(0, 2, 1)
        x_rec = x_rec[:, :self.T, :]

        return x_rec
    
    def forward(self, x):
        """ 完整的前向传播，集成了量化 """
        # 1. 编码
        z_continuous = self.encode(x)
        
        if self.use_vq:
            # 2. 如果启用VQ，进行量化
            z_quantized, indices, vq_loss = self.residual_vq(z_continuous)
            
            # 3. 使用量化后的z进行解码
            x_rec = self.decode(z_quantized)
            
            return x_rec, z_continuous, z_quantized, indices, vq_loss
        else:
            # 如果不使用VQ，则直接解码
            x_rec = self.decode(z_continuous)
            # 为了保持输出格式统一，返回None
            return x_rec, z_continuous, None, None, None