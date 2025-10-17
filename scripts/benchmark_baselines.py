# scripts/benchmark_baselines.py (已修复所有已知错误，特别是归一化问题)
import os
import numpy as np
import yaml
import argparse
import json
import gzip
import math
from scipy.fftpack import dct, idct
from sklearn.decomposition import PCA, MiniBatchDictionaryLearning
import sys
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录（上一级目录）
project_root = os.path.dirname(current_dir)
# 将项目根目录加入 sys.path
sys.path.insert(0, project_root)
# 确保此脚本与 eval_saved.py 在同一查找路径下
try:
    from eval_saved import reconstruct_from_windows_np, inverse_normalize_np, find_json_file_with_keyword
except ImportError:
    print("[错误] 无法导入 'eval_saved.py' 中的辅助函数。请确保此脚本可以访问到它。")
    exit()


def calculate_metrics(orig_final, recon_final):
    """计算所有需要的评估指标"""
    mse = np.mean((orig_final - recon_final) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(orig_final - recon_final))
    signal_power = np.mean(orig_final ** 2)
    noise_power = mse
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    max_val = np.max(orig_final)
    psnr = 10 * np.log10((max_val**2) / noise_power) if noise_power > 0 else float('inf')
    pearson_r = np.corrcoef(orig_final.flatten(), recon_final.flatten())[0, 1]
    return {
        "MSE": float(mse), "RMSE": float(rmse), "MAE": float(mae),
        "SNR_dB": float(snr), "PSNR": float(psnr), "Pearson_r": float(pearson_r)
    }

def benchmark_gzip(orig_sequence_path, out_dir):
    """评估 Gzip"""
    print("\n--- 正在评估 Gzip ---")
    with open(orig_sequence_path, 'rb') as f_in:
        original_data = f_in.read()
    compressed_path = os.path.join(out_dir, "temp_compressed.gz")
    with gzip.open(compressed_path, 'wb') as f_out:
        f_out.write(original_data)
    original_size_bits = len(original_data) * 8
    compressed_size_bits = os.path.getsize(compressed_path) * 8
    compression_ratio = original_size_bits / compressed_size_bits if compressed_size_bits > 0 else float('inf')
    print(f"压缩率: {compression_ratio:.2f}x")
    return {
        "method": "Gzip (Lossless)", "compression_ratio": compression_ratio, "MSE": 0,
        "RMSE": 0, "MAE": 0, "PSNR": float('inf'), "Pearson_r": 1.0
    }


def benchmark_dct(test_windows, orig_sequence, stride, norm_stats_path, keep_coeffs=2):
    """评估变换编码方法 DCT"""
    print(f"\n--- 正在评估 DCT (保留 {keep_coeffs} 个系数) ---")
    N, T, C = test_windows.shape
    test_windows_2d = np.squeeze(test_windows, axis=2) if C >= 1 else test_windows

    rec_windows = np.zeros_like(test_windows_2d)
    for i in range(N):
        window = test_windows_2d[i]
        coeffs = dct(window, type=2, norm='ortho')
        coeffs[keep_coeffs:] = 0
        rec_windows[i] = idct(coeffs, type=2, norm='ortho')

    original_size_bits = N * T * 32
    compressed_size_bits = N * keep_coeffs * 32
    compression_ratio = original_size_bits / compressed_size_bits
    print(f"压缩率 (仅考虑系数): {compression_ratio:.2f}x")

    rec_sequence_normed, _ = reconstruct_from_windows_np(rec_windows, T, stride)
    # **关键修正**: 对重构序列进行逆归一化
    rec_sequence_restored = inverse_normalize_np(rec_sequence_normed, norm_stats_path)

    L = min(len(orig_sequence), len(rec_sequence_restored))
    metrics = calculate_metrics(orig_sequence[:L], rec_sequence_restored[:L])
    return {"method": f"DCT-k{keep_coeffs}", "compression_ratio": compression_ratio, **metrics}


def benchmark_pca(train_windows, test_windows, orig_sequence, stride, norm_stats_path, keep_coeffs=8):
    """评估 PCA (主成分分析)"""
    print(f"\n--- 正在评估 PCA (保留 {keep_coeffs} 个主成分) ---")
    N_train, T, _ = train_windows.shape
    N_test, _, _ = test_windows.shape

    train_windows_2d = np.squeeze(train_windows, axis=2)
    test_windows_2d = np.squeeze(test_windows, axis=2)

    print("正在从训练集学习PCA主成分...")
    pca = PCA(n_components=keep_coeffs)
    pca.fit(train_windows_2d)

    print("正在对测试集进行PCA转换...")
    transformed_windows = pca.transform(test_windows_2d)
    rec_windows = pca.inverse_transform(transformed_windows)

    original_size_bits = N_test * T * 32
    coeffs_size_bits = N_test * keep_coeffs * 32
    components_size_bits = pca.components_.nbytes * 8
    compressed_size_bits = coeffs_size_bits + components_size_bits
    compression_ratio = original_size_bits / compressed_size_bits
    print(f"压缩率 (系数+主成分开销): {compression_ratio:.2f}x")

    rec_sequence_normed, _ = reconstruct_from_windows_np(rec_windows, T, stride)
    # **关键修正**: 对重构序列进行逆归一化
    rec_sequence_restored = inverse_normalize_np(rec_sequence_normed, norm_stats_path)
    
    L = min(len(orig_sequence), len(rec_sequence_restored))
    metrics = calculate_metrics(orig_sequence[:L], rec_sequence_restored[:L])
    return {"method": f"PCA-k{keep_coeffs}", "compression_ratio": compression_ratio, **metrics}


def benchmark_ksvd(train_windows, test_windows, orig_sequence, stride, norm_stats_path, n_atoms=16, sparsity=2):
    """评估 K-SVD (通过 MiniBatchDictionaryLearning 实现)"""
    print(f"\n--- 正在评估 K-SVD (字典大小={n_atoms}, 稀疏度={sparsity}) ---")
    N_train, T, _ = train_windows.shape
    N_test, _, _ = test_windows.shape

    train_windows_2d = np.squeeze(train_windows, axis=2)
    test_windows_2d = np.squeeze(test_windows, axis=2)

    print("正在从训练集学习字典...")
    dico = MiniBatchDictionaryLearning(n_components=n_atoms, alpha=sparsity, max_iter=500, random_state=42)
    dico.fit(train_windows_2d)
    dictionary = dico.components_

    print("正在对测试集进行稀疏编码...")
    sparse_codes = dico.transform(test_windows_2d)
    rec_windows = np.dot(sparse_codes, dictionary)

    original_size_bits = N_test * T * 32
    codes_size_bits = N_test * sparsity * 32
    dictionary_size_bits = n_atoms * T * 32
    print(f"原始数据大小: {original_size_bits} bits,稀疏码大小: {codes_size_bits} bits,字典大小: {dictionary_size_bits} bits")
    compressed_size_bits = codes_size_bits + dictionary_size_bits
    compression_ratio = original_size_bits / compressed_size_bits
    print(f"压缩率 (稀疏码+字典): {compression_ratio:.2f}x")

    rec_sequence_normed, _ = reconstruct_from_windows_np(rec_windows, T, stride)
    # **关键修正**: 对重构序列进行逆归一化
    rec_sequence_restored = inverse_normalize_np(rec_sequence_normed, norm_stats_path)

    L = min(len(orig_sequence), len(rec_sequence_restored))
    metrics = calculate_metrics(orig_sequence[:L], rec_sequence_restored[:L])
    return {"method": f"K-SVD-d{n_atoms}-s{sparsity}", "compression_ratio": compression_ratio, **metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark baseline compression methods.")
    parser.add_argument("--cfg", required=True, help="Path to the config file for your dataset.")
    parser.add_argument("--out_dir", type=str, default="results_benchmark", help="Directory to save benchmark results.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.cfg, encoding='utf-8'))
    os.makedirs(args.out_dir, exist_ok=True)

    # --- 1. 加载并准备所有数据 ---
    train_windows = np.load(cfg['data']['train_np'])
    test_windows = np.load(cfg['data']['test_np'])
    stride = cfg.get('data', {}).get('stride', 16)
    
    orig_sequence_from_windows, _ = reconstruct_from_windows_np(test_windows, test_windows.shape[1], stride)
    prepared_dir = os.path.dirname(cfg['data']['test_np'])
    norm_stats_path = find_json_file_with_keyword(prepared_dir, 'norm_stats')
    if norm_stats_path:
        final_orig_sequence = inverse_normalize_np(orig_sequence_from_windows, norm_stats_path)
    else:
        final_orig_sequence = orig_sequence_from_windows
    temp_orig_path = os.path.join(args.out_dir, "temp_orig_sequence.npy")
    np.save(temp_orig_path, final_orig_sequence.astype(np.float32))
    
    
    # --- 2. 运行所有基准测试 ---
    results = []
    results.append(benchmark_gzip(temp_orig_path, args.out_dir))
    
    results.append(benchmark_dct(test_windows, final_orig_sequence, stride, norm_stats_path, keep_coeffs=8))
    results.append(benchmark_dct(test_windows, final_orig_sequence, stride, norm_stats_path, keep_coeffs=2))
    
    results.append(benchmark_pca(train_windows, test_windows, final_orig_sequence, stride, norm_stats_path, keep_coeffs=8))
    results.append(benchmark_pca(train_windows, test_windows, final_orig_sequence, stride, norm_stats_path, keep_coeffs=2))

    results.append(benchmark_ksvd(train_windows, test_windows, final_orig_sequence, stride, norm_stats_path, n_atoms=128, sparsity=4))
    results.append(benchmark_ksvd(train_windows, test_windows, final_orig_sequence, stride, norm_stats_path, n_atoms=256, sparsity=2))

    # --- 3. 打印和保存结果 ---
    print("\n\n--- 最终基准测试结果汇总 ---")
    print(f"{'Method':<25} | {'CR':>6} | {'PSNR':>8} | {'Pearson_r':>10} | {'RMSE':>8}")
    print("-" * 70)
    for res in results:
        print(f"{res['method']:<25} | {res.get('compression_ratio', 0):>6.2f}x | {res.get('PSNR', 0):>8.2f} | {res.get('Pearson_r', 0):>10.4f} | {res.get('RMSE', 0):>8.4f}")
        
    summary_path = os.path.join(args.out_dir, "benchmark_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n详细结果已保存至: {summary_path}")