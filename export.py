# export.py
import torch, yaml
from models.cnn_ae import CNNAutoEncoder

cfg = yaml.safe_load(open("configs/config.yaml"))
ckpt = torch.load(cfg['train']['ckpt'], map_location='cpu')
T, C = cfg['data']['T'], cfg['data']['C']
model = CNNAutoEncoder(T=T, C=C, latent_dim=cfg['model']['latent_dim'])
model.load_state_dict(ckpt['model'])
model.eval()
# Trace with dummy
import torch
dummy = torch.randn(1, T, C)
traced = torch.jit.trace(model, dummy)
traced.save("model_traced.pt")
print("Exported model_traced.pt")
