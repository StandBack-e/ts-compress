#!/usr/bin/env python3
"""
evaluate_restoration.py (Updated)

用途:
  - 采用最严谨的方式，对模型在测试集上的性能进行最终评估。
  - 加载 test_orig_seq.npy 和 test_rec_seq.npy。
  - 对这两个序列应用完全相同的逆向变换。
  - 比较还原后的序列，并计算更全面的指标 (MSE, RMSE, MAE, SNR, PSNR, Pearson r)。
  - 加入了边界不连续性得分 (BDS) 的计算。

用法示例:
python scripts/evaluate_restoration.py `
  --orig_seq results/recon_full_w32/test_orig_seq.npy `
  --rec_seq results/recon_full_w32/test_rec_seq.npy `
  --prepared_dir data/prepared/FD001_w32_s16_norm_detrend `
  --out results/final_restored_comparison
"""
import os
import json
import argparse
import re # 新增：用于从目录名中提取stride
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =========== 开始添加指标计算函数 ============
def mse(a,b): return float(((a-b)**2).mean())
def rmse(a,b): return float(np.sqrt(((a-b)**2).mean()))
def mae(a,b): return float(np.mean(np.abs(a-b)))
def psnr(a,b, data_range=None):
    msev = ((a-b)**2).mean()
    if msev == 0: return float('inf')
    if data_range is None:
        data_range = a.max() - a.min()
        if data_range == 0: data_range = 1.0
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(msev))

def snr(a,b):
    signal_power = (a**2).mean()
    noise_power = ((a-b)**2).mean()
    if noise_power==0: return float('inf')
    return float(10.0 * np.log10(signal_power / noise_power))
# =========== 结束添加 ============

# =========== 新增：BDS指标计算函数 ============
def calculate_bds(sequence, stride):
    """计算边界不连续性得分 (Boundary Discontinuity Score)"""
    sequence = np.asarray(sequence).squeeze()
    discontinuities = []
    # 从第一个接缝点开始，遍历所有接缝点
    for i in range(stride, len(sequence), stride):
        # 计算接缝点与其前一个点的差值的平方
        jump = (sequence[i] - sequence[i-1])**2
        discontinuities.append(jump)
    
    if not discontinuities:
        return 0.0
        
    return float(np.mean(discontinuities))
# ============================================

# =========== 新增：从目录名推断stride的函数 ============
def infer_stride_from_dirname(prepared_dir):
    """从 prepared_dir 的名称中解析出 stride """
    basename = os.path.basename(prepared_dir)
    # 尝试匹配 'w32_s16' 这样的格式
    m = re.search(r'w\d+_s(\d+)', basename)
    if m:
        return int(m.group(1))
    # 尝试匹配 '_s16' 这样的格式
    m2 = re.search(r'_s(\d+)', basename)
    if m2:
        return int(m2.group(1))
    print("[warn] Could not infer stride from directory name. BDS calculation might be incorrect.")
    return None # 如果找不到，返回None
# ================================================

def find_json_file(directory, keyword):
    """在目录中查找包含特定关键词的json文件"""
    for fname in os.listdir(directory):
        if keyword in fname and fname.endswith('.json'):
            return os.path.join(directory, fname)
    return None

def restore_sequence(seq, prepared_dir):
    """对单个序列执行完整的逆向变换（逆归一化 + 逆去趋势）"""
    
    # --- 步骤 1: 逆归一化 ---
    norm_stats_path = find_json_file(prepared_dir, 'norm_stats')
    if norm_stats_path and os.path.exists(norm_stats_path):
        with open(norm_stats_path) as f:
            stats = json.load(f)
        mean = float(stats['mean'])
        scale = float(stats['scale'])
        seq = seq * scale + mean
        print(f"[restore] Applied inverse normalization using: {os.path.basename(norm_stats_path)}")
    else:
        print("[restore] No norm_stats file found. Skipping normalization.")

    # --- 步骤 2: 逆去趋势 (添加趋势) ---
    trend_params_path = find_json_file(prepared_dir, 'trend_params')
    if trend_params_path and os.path.exists(trend_params_path):
        with open(trend_params_path) as f:
            tp = json.load(f)

        full_len = tp.get('full_series_length')
        if full_len is None:
            raise ValueError("'full_series_length' not found in trend_params.json.")

        slope = tp.get('slope', 0.0)
        intercept = tp.get('intercept', 0.0)
        
        L = len(seq)
        t = np.arange(full_len - L, full_len)
        trend = slope * t + intercept
        
        seq = seq + trend
        print(f"[restore] Applied inverse detrending using: {os.path.basename(trend_params_path)}")
    else:
        print("[restore] No trend_params file found. Skipping detrending.")
        
    return seq


def main(args):
    os.makedirs(args.out, exist_ok=True)

    orig_seq_processed = np.load(args.orig_seq).squeeze()
    rec_seq_processed = np.load(args.rec_seq).squeeze()
    
    print("\n--- Restoring ORIGINAL sequence ---")
    orig_seq_restored = restore_sequence(orig_seq_processed, args.prepared_dir)
    
    print("\n--- Restoring RECONSTRUCTED sequence ---")
    rec_seq_restored = restore_sequence(rec_seq_processed, args.prepared_dir)
    
    print(f"\n[info] Restoration complete. Final length for comparison: {len(orig_seq_restored)}")

    # =========== 开始修改指标计算部分 ============
    metrics = {
        "description": "Comparison between restored original and reconstructed sequences."
    }
    a, b = orig_seq_restored, rec_seq_restored
    metrics['MSE'] = mse(a, b)
    metrics['RMSE'] = rmse(a, b)
    metrics['MAE'] = mae(a, b)
    metrics['SNR_dB'] = snr(a, b)
    metrics['PSNR'] = psnr(a, b)
    metrics['Pearson_r'] = float(stats.pearsonr(a, b)[0])
    metrics['comparison_length'] = len(a)
    # =========== 结束修改 ============

     # =========== 新增：计算并添加BDS指标 ============
    stride = infer_stride_from_dirname(args.prepared_dir)
    if stride is not None:
        bds_score_orig = calculate_bds(a, stride)
        bds_score_rec = calculate_bds(b, stride)
        metrics['BDS_original'] = bds_score_orig
        metrics['BDS_reconstructed'] = bds_score_rec
        print(f"[info] BDS calculated with stride={stride}: Original={bds_score_orig:.6f}, Reconstructed={bds_score_rec:.6f}")
    # =============================================
    
    metrics_path = os.path.join(args.out, "restoration_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[info] Saved final restoration metrics -> {metrics_path}")
    print(json.dumps(metrics, indent=2))

    # ... (绘图部分不变)
    plt.figure(figsize=(16, 4))
    plt.plot(orig_seq_restored, label='Original Sequence (Restored)', linewidth=1.5, color='blue', alpha=0.9)
    plt.plot(rec_seq_restored, label='Reconstructed Sequence (Restored)', linewidth=1.5, color='red', alpha=0.8, linestyle='--')
    plt.legend()
    plt.title("Final Comparison: Restored Original vs. Restored Reconstructed Sequence")
    plt.xlabel("Time Step")
    plt.ylabel("Sensor Value (Original Scale)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plot_path = os.path.join(args.out, "final_restored_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[info] Saved comparison plot -> {plot_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Restore and compare original vs reconstructed sequences for final evaluation.")
    p.add_argument("--orig_seq", required=True, help="Path to test_orig_seq.npy (ground truth windows reassembled)")
    p.add_argument("--rec_seq", required=True, help="Path to test_rec_seq.npy (reconstructed windows reassembled)")
    p.add_argument("--prepared_dir", required=True, help="Path to the 'prepared' directory with norm_stats and trend_params")
    p.add_argument("--out", required=True, help="Output directory for final metrics and plots")
    args = p.parse_args()
    main(args)