# train.py
import argparse
import yaml
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from data.loaders import get_dataloaders
from models.cnn_ae import CNNAutoEncoder
from models.cnn_rnn_ae import CNNRNNAutoEncoder
from models.controller import ComplexityEstimator
from utils import weighted_mse, extract_shallow_features
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main(cfg):
    # 简单配置加载
    print(torch.cuda.is_available())
    # exit(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # load data arrays (你需要准备 train.npy val.npy)
    train_arr = np.load(cfg['data']['train_np'])
    val_arr = np.load(cfg['data']['val_np'])
    print(f"Train shape: {train_arr.shape} Val shape: {val_arr.shape}")
    T = train_arr.shape[1]
    C = train_arr.shape[2]
    

    train_loader, val_loader = get_dataloaders(train_arr, val_arr, batch_size=cfg['train']['batch_size'])

    # model selection
    if cfg['model']['type'] == 'cnn':
        model = CNNAutoEncoder(T=T, C=C, latent_dim=cfg['model']['latent_dim']).to(device)
    else:
        model = CNNRNNAutoEncoder(T=T, C=C, latent_dim=cfg['model']['latent_dim']).to(device)

    controller = ComplexityEstimator(in_dim=5, hidden=64, out_scale=1.0).to(device)

    opt = optim.Adam(list(model.parameters()) + list(controller.parameters()), lr=float(cfg['train']['lr']))
    scheduler = ReduceLROnPlateau(opt, 'min', patience=5, factor=0.5)

    best_val = 1e9

    for epoch in range(cfg['train']['epochs']):
        model.train(); controller.train()
        total_loss = 0
        for xb in train_loader:
            xb = xb.to(device)
            # controller features from xb
            feats = extract_shallow_features(xb)  # numpy
            feats = torch.from_numpy(feats).to(device)
            ratio = controller(feats)  # (B,)
            # scale latent: 这里示意性使用，实际可以通过 gating / pruning / quantization 实现
            rec, z = model(xb)
            # 1. 原始的重构损失 (MSE)
            recon_loss_mse = weighted_mse(xb, rec)

            # 2. 计算频域损失 (STFT Loss)
            # permute a (B, T, 1) tensor to (B, T) for stft
            xb_squeeze = xb.squeeze(-1)
            rec_squeeze = rec.squeeze(-1)
            
            # n_fft: 窗口大小, hop_length: 步长, win_length: 窗函数长度
            # 这些是超参数，可以根据你的数据特性调整
            stft_orig = torch.stft(xb_squeeze, n_fft=32, hop_length=8, win_length=32, return_complex=True)
            stft_rec = torch.stft(rec_squeeze, n_fft=32, hop_length=8, win_length=32, return_complex=True)
            
            # 计算STFT幅度谱的L1损失 (通常比L2更鲁棒)
            stft_loss = F.l1_loss(torch.abs(stft_rec), torch.abs(stft_orig))

            # 3. 组合损失
            # alpha 是一个超参数，用于平衡时域和频域损失的重要性，可以从 0.1, 0.5, 1.0 开始尝试
            alpha = 0.0
            # print(f"recon_loss_mse: {recon_loss_mse.item():.6f}, stft_loss: {stft_loss.item():.6f}")
            recon_loss = recon_loss_mse + alpha * stft_loss

            # 4. 加上原有的潜在空间惩罚
            # penalize latent size scaled by ratio mean (鼓励更小 latent)
            latent_penalty = z.abs().mean() * ratio.mean()
            loss = recon_loss + float(cfg['train']['lambda_latent']) * latent_penalty
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        # val
        model.eval(); controller.eval()
        val_loss = 0
        with torch.no_grad():
            for xb in val_loader:
                xb = xb.to(device)
                rec, z = model(xb)
                val_loss += weighted_mse(xb, rec).item()
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
    cfg = yaml.safe_load(open(args.cfg))
    main(cfg)
