# evaluate_quantized.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import json
import csv
import re
import torch
import numpy as np
import math
from torch.utils.data import DataLoader
# 确保引用您项目中的正确模块
from data.loaders import TimeSeriesDataset
from models.cnn_rnn_attention_ae import CNNRNNAttentionAutoEncoder 
# 引用 eval_saved.py 中的辅助函数
from eval_saved import reconstruct_from_windows_np, inverse_normalize_np, add_linear_trend_np, find_json_file_with_keyword, infer_stride_from_dirname

def evaluate_quantized_and_save(cfg, out_dir="results_quantized"):
    """
    专门用于评估量化模型的脚本。
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. 加载数据 ---
    test_arr = np.load(cfg['data']['test_np'])
    ds = TimeSeriesDataset(test_arr)
    # 使用 batch_size=cfg['train']['batch_size'] 以加速推理
    dl = DataLoader(ds, batch_size=cfg['train']['batch_size'], shuffle=False)
    
    T, C = test_arr.shape[1], test_arr.shape[2]
    num_windows = len(test_arr)

    # --- 2. 加载训练好的量化模型 ---
    # 确保 config.yaml 中 use_vq 设置为 true
    if not cfg['model'].get('use_vq', False):
        raise ValueError("请在您的配置文件中设置 'use_vq: true' 来评估量化模型！")

    # 从配置文件中获取 num_quantizers 和 codebook_size
    num_quantizers = cfg['model'].get('num_quantizers', 4)
    codebook_size = cfg['model'].get('codebook_size', 256)

    model = CNNRNNAttentionAutoEncoder(
        T=T, C=C,
        latent_dim=cfg['model']['latent_dim'],
        use_vq=True,
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
    ).to(device)

    checkpoint = torch.load(cfg['train']['ckpt'], map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print("量化模型加载成功!")

    # --- 3. 逐批次进行推理，获取重构窗口和量化索引 ---
    all_orig_windows = []
    all_rec_windows = []
    all_indices = []

    with torch.no_grad():
        for xb in dl:
            xb = xb.to(device).float()
            
            # 完整前向传播
            # forward返回: x_rec, z_continuous, z_quantized, indices, vq_loss
            rec, _, _, indices, _ = model(xb)
            
            all_orig_windows.append(xb.cpu().numpy())
            all_rec_windows.append(rec.cpu().numpy())
            if indices is not None:
                all_indices.append(indices.cpu().numpy())

    all_orig_windows = np.concatenate(all_orig_windows, axis=0)
    all_rec_windows = np.concatenate(all_rec_windows, axis=0)
    all_indices = np.concatenate(all_indices, axis=0) # shape: [N, num_quantizers]

    # --- 4. 保存压缩结果 (索引 + 码本) ---
    # 4.1 保存量化索引
    indices_path = os.path.join(out_dir, "quantized_indices.npy")
    np.save(indices_path, all_indices)
    print(f"量化索引已保存至: {indices_path}")

    # 4.2 保存码本
    codebooks = []
    for quantizer in model.residual_vq.quantizers:
        codebooks.append(quantizer.embedding.weight.data.cpu().numpy())
    
    codebook_path = os.path.join(out_dir, "codebooks.npy")
    np.save(codebook_path, np.array(codebooks, dtype=object)) # 保存为对象数组以处理可能的维度不一致
    print(f"码本已保存至: {codebook_path}")

    # --- 5. 精确计算压缩率 (核心修正部分) ---
    print("\n--- 压缩率报告 (基于测试集精确计算) ---")

    # 5.1 计算与测试集(test.npy)对应的真实原始数据大小 (in bits)
    # 我们不再使用 full_raw.npy，因为我们只在 test.npy 上进行了评估
    stride = cfg.get('data', {}).get('stride', 16) # 从配置获取 stride
    num_windows_in_test = len(test_arr) # test_arr 就是 test.npy
    T = test_arr.shape[1] # 窗口大小
    C = test_arr.shape[2] # 通道数

    # 精确计算 test.npy 所代表的去重叠后的序列长度
    if num_windows_in_test > 0:
        num_points_in_test_sequence = (num_windows_in_test - 1) * stride + T
    else:
        num_points_in_test_sequence = 0

    # 计算原始大小 (bits)，假设原始数据为 float32
    original_size_bits = num_points_in_test_sequence * C * 32
    print(f"测试集窗口数: {num_windows_in_test}")
    print(f"计算出的测试集对应原始序列长度: {num_points_in_test_sequence}")
    print(f"对应的原始数据大小: {original_size_bits / 8 / 1024:.2f} KB")
    
    # 5.2 压缩后数据大小 (in bits)
    num_quantizers = cfg['model']['num_quantizers']
    codebook_size = cfg['model']['codebook_size']
    latent_dim = cfg['model']['latent_dim']
    bits_per_index = math.ceil(math.log2(codebook_size))
    
    # a. 索引大小
    indices_size_bits = all_indices.size * bits_per_index
    
    # b. 码本大小 (固定开销)
    codebook_size_bits = num_quantizers * codebook_size * latent_dim * 32 # float32
    
    # c. 压缩后总大小
    compressed_size_bits = indices_size_bits
    
    # d. 计算压缩率
    compression_ratio = original_size_bits / compressed_size_bits
    
    print("\n--- 压缩率报告 ---")
    print(f"原始数据大小: {original_size_bits / 8 / 1024:.2f} KB")
    print(f"压缩后大小 (索引 + 码本): {compressed_size_bits / 8 / 1024:.2f} KB")
    print(f"  - 索引部分: {indices_size_bits / 8 / 1024:.2f} KB")
    print(f"  - 码本部分 (固定开销): {codebook_size_bits / 8 / 1024:.2f} KB")
    print(f"量化后压缩率: {compression_ratio:.2f} 倍\n")


    # --- 6. 序列重构与后处理 (与 eval_saved.py 逻辑相同) ---
    stride = cfg.get('data', {}).get('stride', 16) # 从配置获取 stride
    
    recon_seq_normed, coverage = reconstruct_from_windows_np(all_rec_windows, T, stride)
    orig_seq_normed, _ = reconstruct_from_windows_np(all_orig_windows, T, stride)

    prepared_dir = os.path.dirname(cfg['data']['test_np'])
    norm_stats_path = find_json_file_with_keyword(prepared_dir, 'norm_stats')
    
    if norm_stats_path:
        recon_seq_restored = inverse_normalize_np(recon_seq_normed, norm_stats_path)
        orig_seq_restored = inverse_normalize_np(orig_seq_normed, norm_stats_path)
    else:
        recon_seq_restored = recon_seq_normed
        orig_seq_restored = orig_seq_normed

    # --- 7. 计算所有评估指标 ---
    L = min(len(orig_seq_restored), len(recon_seq_restored))
    orig_final = orig_seq_restored[:L]
    recon_final = recon_seq_restored[:L]

    mse = np.mean((orig_final - recon_final) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(orig_final - recon_final))
    signal_power = np.mean(orig_final ** 2)
    noise_power = mse
    snr = 10 * np.log10(signal_power / noise_power)
    
    max_val = np.max(orig_final)
    psnr = 10 * np.log10((max_val**2) / mse)
    
    # 计算皮尔逊相关系数
    if C > 1:
        corrs = [np.corrcoef(orig_final[:, i], recon_final[:, i])[0, 1] for i in range(C)]
        pearson_r = np.mean(corrs)
    else:
        pearson_r = np.corrcoef(orig_final, recon_final)[0, 1]

    # --- 8. 保存summary.json文件 ---
    summary = {
        "description": "Evaluation of the Quantized Compression Model.",
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "SNR_dB": float(snr),
        "PSNR": float(psnr),
        "Pearson_r": float(pearson_r),
        "compression_ratio": float(compression_ratio),
        "original_size_KB": original_size_bits / 8 / 1024,
        "compressed_size_KB": compressed_size_bits / 8 / 1024,
        "indices_size_KB": indices_size_bits / 8 / 1024,
        "codebook_size_KB": codebook_size_bits / 8 / 1024,
    }
    with open(os.path.join(out_dir,"summary_quantized.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"评估结果已保存至: {os.path.join(out_dir, 'summary_quantized.json')}")

    # --- 9. 绘图 ---
    plt.figure(figsize=(15, 4))
    plt.plot(orig_final, label='Original Restored', color='blue')
    plt.plot(recon_final, label='Reconstructed from Quantized', color='red', alpha=0.9)
    plt.legend()
    plt.title(f"Final Restored Comparison (Quantized) - PSNR: {psnr:.2f} dB, CR: {compression_ratio:.2f}x")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "final_restored_comparison_quantized.png"), dpi=300)
    plt.close()
    print(f"对比图已保存至: {os.path.join(out_dir, 'final_restored_comparison_quantized.png')}")


if __name__ == "__main__":
    import yaml, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to the config file for the trained quantized model.")
    parser.add_argument("--out_dir", type=str, default="results_quantized", help="Directory to save evaluation results.")
    args = parser.parse_args()
    
    cfg = yaml.safe_load(open(args.cfg, encoding='utf-8'))
    
    # 自动在配置文件名后加上传感器信息和"_quantized"后缀来创建输出目录
    # basename = os.path.splitext(os.path.basename(args.cfg))[0]
    # out_dir_named = f"results/{basename}_quantized"

    out_dir_named = args.out_dir

    evaluate_quantized_and_save(cfg, out_dir=out_dir_named)