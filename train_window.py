# train.py (修改后),每次给两个连续的滑动窗口来训练
import argparse
import yaml
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from data.loader_window import get_dataloaders
from models.cnn_ae import CNNAutoEncoder
from models.cnn_rnn_ae import CNNRNNAutoEncoder
from models.controller import ComplexityEstimator
from utils import weighted_mse, extract_shallow_features
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_arr = np.load(cfg['data']['train_np'])
    val_arr = np.load(cfg['data']['val_np'])
    print(f"Train shape: {train_arr.shape} Val shape: {val_arr.shape}")
    T = train_arr.shape[1]
    C = train_arr.shape[2]
    
    # 从配置中获取 stride，用于计算重叠长度
    # 假设 stride 在 config.yaml 的 data 部分
    stride = cfg['data']['stride']
    overlap_len = T - stride
    if overlap_len <= 0:
        raise ValueError("Window size must be greater than stride to have overlap.")
    print(f"Window={T}, Stride={stride}, Overlap length for consistency loss={overlap_len}")

    train_loader, val_loader = get_dataloaders(train_arr, val_arr, batch_size=cfg['train']['batch_size'])

    if cfg['model']['type'] == 'cnn':
        model = CNNAutoEncoder(T=T, C=C, latent_dim=cfg['model']['latent_dim']).to(device)
    else:
        model = CNNRNNAutoEncoder(T=T, C=C, latent_dim=cfg['model']['latent_dim']).to(device)

    controller = ComplexityEstimator(in_dim=5, hidden=64, out_scale=1.0).to(device)
    params_to_optimize = list(model.parameters()) + list(controller.parameters())
    opt = optim.Adam(params_to_optimize, lr=float(cfg['train']['lr']))  
    
    scheduler = ReduceLROnPlateau(opt, 'min', patience=5, factor=0.5)
    best_val = 1e9

    # 从配置中获取超参数
    lambda_latent = float(cfg['train']['lambda_latent'])
    # lambda_consistency = float(cfg['train']['lambda_consistency'])
    # 新的潜在空间一致性损失权重
    lambda_latent_consistency = float(cfg['train']['lambda_latent_consistency'])
    alpha_stft = float(cfg['train']['alpha_stft'])

    for epoch in range(cfg['train']['epochs']):
        model.train(); controller.train()
        total_loss = 0
        
        # 修改训练循环以处理成对的窗口
        for xb1, xb2 in train_loader:
            xb1 = xb1.to(device)
            xb2 = xb2.to(device)

            # # 分别对两个窗口进行重构
            # rec1, z1 = model(xb1)
            # rec2, z2 = model(xb2)

            # # --- 1. 计算平均重构损失 (MSE + STFT) ---
            # # MSE
            # recon_loss_mse1 = weighted_mse(xb1, rec1)
            # recon_loss_mse2 = weighted_mse(xb2, rec2)
            # avg_recon_loss_mse = (recon_loss_mse1 + recon_loss_mse2) / 2.0
            
            # # STFT
            # xb1_s = xb1.squeeze(-1); rec1_s = rec1.squeeze(-1)
            # xb2_s = xb2.squeeze(-1); rec2_s = rec2.squeeze(-1)
            # stft_orig1 = torch.stft(xb1_s, n_fft=32, hop_length=8, win_length=32, return_complex=True)
            # stft_rec1 = torch.stft(rec1_s, n_fft=32, hop_length=8, win_length=32, return_complex=True)
            # stft_orig2 = torch.stft(xb2_s, n_fft=32, hop_length=8, win_length=32, return_complex=True)
            # stft_rec2 = torch.stft(rec2_s, n_fft=32, hop_length=8, win_length=32, return_complex=True)
            
            # stft_loss1 = F.l1_loss(torch.abs(stft_rec1), torch.abs(stft_orig1))
            # stft_loss2 = F.l1_loss(torch.abs(stft_rec2), torch.abs(stft_orig2))
            # avg_stft_loss = (stft_loss1 + stft_loss2) / 2.0

            # # 组合重构损失
            # total_recon_loss = avg_recon_loss_mse + alpha_stft * avg_stft_loss

            # # --- 2. 计算潜在空间惩罚 ---
            # # 这里可以取平均，或者合并处理，取平均更简单
            # avg_z_abs_mean = (z1.abs().mean() + z2.abs().mean()) / 2.0
            # # controller features (只用第一个窗口的特征作为示例)
            # feats = extract_shallow_features(xb1)
            # feats = torch.from_numpy(feats).to(device)
            # ratio = controller(feats)
            # latent_penalty = avg_z_abs_mean * ratio.mean()

            # # --- 3. 计算重叠区域一致性损失 (Overlap Consistency Loss) ---
            # # rec1的后半部分重叠区域
            # overlap1 = rec1[:, -overlap_len:, :]
            # # rec2的前半部分重叠区域
            # overlap2 = rec2[:, :overlap_len, :]
            # consistency_loss = F.mse_loss(overlap1, overlap2)
            
            # # --- 4. 组合最终的总损失 ---
            # # print(f"consistency_loss={consistency_loss.item():.6f}")
            # loss = total_recon_loss + lambda_latent * latent_penalty + lambda_consistency * consistency_loss
            # 1. 编码得到潜在向量
            z1 = model(xb1)
            z2 = model(xb2)

            # 2. 解码得到重构结果
            rec1 = model.decode(z1)
            rec2 = model.decode(z2)

            # --- 计算重构损失 (与之前类似，取平均) ---
            recon_loss_mse1 = weighted_mse(xb1, rec1)
            recon_loss_mse2 = weighted_mse(xb2, rec2)
            avg_recon_loss_mse = (recon_loss_mse1 + recon_loss_mse2) / 2.0
            
            xb1_s, rec1_s = xb1.squeeze(-1), rec1.squeeze(-1)
            xb2_s, rec2_s = xb2.squeeze(-1), rec2.squeeze(-1)
            # 定义STFT参数
            n_fft = 32
            hop_length = 8
            win_length = 32

            # 创建一个汉宁窗
            window = torch.hann_window(win_length, device=device)
            stft_orig1 = torch.stft(xb1_s, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
            stft_rec1 = torch.stft(rec1_s, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
            stft_orig2 = torch.stft(xb2_s, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
            stft_rec2 = torch.stft(rec2_s, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
            stft_loss1 = F.l1_loss(torch.abs(stft_rec1), torch.abs(stft_orig1))
            stft_loss2 = F.l1_loss(torch.abs(stft_rec2), torch.abs(stft_orig2))
            avg_stft_loss = (stft_loss1 + stft_loss2) / 2.0
            total_recon_loss = avg_recon_loss_mse + alpha_stft * avg_stft_loss

            # --- 计算潜在空间惩罚 ---
            avg_z_abs_mean = (z1.abs().mean() + z2.abs().mean()) / 2.0
            feats = extract_shallow_features(xb1)
            feats = torch.from_numpy(feats).to(device)
            ratio = controller(feats)
            latent_penalty = avg_z_abs_mean * ratio.mean()

            # --- 3. 计算新的潜在空间预测损失 ---
            z2_predicted = model.latent_predictor(z1)
            # 目标是让 z1 预测出的 z2' 逼近真实的 z2
            # 使用 .detach() 是一个好的实践，它确保这个损失只更新 predictor 和 encoder，
            # 而不会让 z2 的梯度“回头”影响 encoder 两次。
            latent_consistency_loss = F.mse_loss(z2_predicted, z2.detach())
            
            # --- 4. 组合最终的总损失 ---
            loss = total_recon_loss + lambda_latent * latent_penalty + lambda_latent_consistency * latent_consistency_loss
            
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        # 验证循环 (保持简单，只评估核心的重构MSE)
        model.eval(); controller.eval()
        val_loss = 0
        with torch.no_grad():
            for xb1, xb2 in val_loader:
                xb1, xb2 = xb1.to(device), xb2.to(device)
                rec1 = model.decode(model(xb1))
                rec2 = model.decode(model(xb2))
                # 评估两个窗口的平均MSE
                val_loss += ((weighted_mse(xb1, rec1) + weighted_mse(xb2, rec2)) / 2.0).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        print(f"Epoch {epoch} train_loss {total_loss/len(train_loader):.6f} val_loss {val_loss:.6f}")
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model': model.state_dict(), 'controller': controller.state_dict()}, cfg['train']['ckpt'])
            
    print("train done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg,encoding='utf-8'))
    main(cfg)