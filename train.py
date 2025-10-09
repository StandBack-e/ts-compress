# train.py
import argparse
import yaml
import numpy as np
import torch
from torch import optim
from data.loaders import get_dataloaders
from models.cnn_ae import CNNAutoEncoder
from models.cnn_rnn_ae import CNNRNNAutoEncoder
from models.controller import ComplexityEstimator
from utils import weighted_mse, extract_shallow_features

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
            recon_loss = weighted_mse(xb, rec)
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
