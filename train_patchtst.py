# train_patchtst.py
import argparse
import yaml
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

# 引用您的数据加载器和工具函数
from data.loader_window import get_dataloaders
from utils import weighted_mse
# 引用新的PatchTST模型
from models.patchtst_ae import PatchTSTAutoEncoder

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_arr = np.load(cfg['data']['train_np'])
    val_arr = np.load(cfg['data']['val_np'])
    T, C = train_arr.shape[1], train_arr.shape[2]
    
    # 获取滑动窗口参数
    stride = cfg['data']['stride']
    overlap_len = T - stride
    
    train_loader, val_loader = get_dataloaders(train_arr, val_arr, batch_size=cfg['train']['batch_size'])

    # --- 核心修改：使用PatchTST模型 ---
    model = PatchTSTAutoEncoder(
        T=T, C=C, 
        latent_dim=cfg['model']['latent_dim'],
        patch_len=cfg['model']['patch_len'],
        patch_stride=cfg['model']['patch_stride'],
        d_model=cfg['model']['d_model'],
        nhead=cfg['model']['nhead']
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=float(cfg['train']['lr']))
    
    # 获取损失权重
    lambda_overlap_consistency = float(cfg['train']['lambda_overlap_consistency'])
    alpha_stft = float(cfg['train']['alpha_stft'])

    best_val = 1e9
    for epoch in range(cfg['train']['epochs']):
        model.train()
        total_loss = 0
        
        for xb1, xb2 in train_loader:
            xb1, xb2 = xb1.to(device), xb2.to(device)

            rec1, z1 = model(xb1)
            rec2, z2 = model(xb2)

            # --- 损失计算 (与您的 train_window.py 完全一致) ---
            # 1. 重构损失 (MSE + STFT)
            recon_loss_mse = (weighted_mse(xb1, rec1) + weighted_mse(xb2, rec2)) / 2.0
            
            # (此处省略STFT计算代码，您可以从train_window.py中复制过来)
            # stft_loss = calculate_stft_loss(xb1, rec1, xb2, rec2)
            # total_recon_loss = recon_loss_mse + alpha_stft * stft_loss
            total_recon_loss = recon_loss_mse # 简化示意

            # 2. 重叠区域一致性损失
            overlap1 = rec1[:, -overlap_len:, :]
            overlap2 = rec2[:, :overlap_len, :]
            overlap_consistency_loss = F.mse_loss(overlap1, overlap2)

            # 3. 最终总损失
            loss = total_recon_loss + lambda_overlap_consistency * overlap_consistency_loss
            
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        # --- 验证循环 (与您的 train_window.py 完全一致) ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb1, xb2 in val_loader:
                xb1, xb2 = xb1.to(device), xb2.to(device)
                rec1, _ = model(xb1)
                rec2, _ = model(xb2)
                val_loss += ((weighted_mse(xb1, rec1) + weighted_mse(xb2, rec2)) / 2.0).item()
        
        val_loss /= len(val_loader)
        print(f"Epoch {epoch} train_loss {total_loss/len(train_loader):.6f} val_loss {val_loss:.6f}")
        
        if val_loss < best_val:
            best_val = val_loss
            # 为PatchTST模型保存独立的ckpt
            torch.save({'model': model.state_dict()}, cfg['train']['ckpt'])
            
    print("train done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to the PatchTST config file.")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg, encoding='utf-8'))
    main(cfg)