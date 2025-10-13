# models/cnn_rnn_ae.py
import torch
import torch.nn as nn

class CNNRNNAttentionAutoEncoder(nn.Module):
    """
    一个结合了CNN、RNN和Attention的混合自编码器模型。
    - CNN部分用于从时间序列中提取局部的、空间上的特征。
    - RNN部分用于学习这些特征在时间上的依赖关系。
    """
    def __init__(self, T, C, latent_dim=16, base_channels=32, rnn_hidden_dim=64, nhead=4):
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
        cnn_out_seq_len = T // 4
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
            # 输出: (B, C, T*4)，这里假设T能被4整除
        )

    def forward(self, x):
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
        _, h_n = self.rnn_encoder(cnn_out)
        
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