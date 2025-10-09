#!/usr/bin/env python3
"""
compare_full_raw_vs_rec.py

用途：
  - 将原始完整序列 (full_raw.npy) 与 eval_saved 生成的 test_rec_seq.npy 做对比
  - 计算 MSE / RMSE / MAE / PSNR / SNR / Pearson r 等指标
  - 画全局对比图、误差时间序列、局部放大图、误差直方图
  - 支持可选的互相关对齐（自动搜索最佳位移）

用法示例：
python scripts/compare_full_raw_vs_rec.py `
  --full_raw data/prepared/FD001_w32_s16_norm_detrend/cmapss_w32_s16_norm_detrend_full_raw.npy `
  --rec results/recon_full_w32/test_rec_seq.npy `
  --out results/compare_full_raw_vs_rec `
  --align xcorr

如果你确定两序列已对齐，去掉 --align 参数即可（默认裁剪到最短）。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 若你采用应急办法
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse, numpy as np
# import matplotlib.pyplot as plt
from scipy import signal, stats

def mse(a,b): return float(((a-b)**2).mean())
def rmse(a,b): return float(np.sqrt(((a-b)**2).mean()))
def mae(a,b): return float(np.mean(np.abs(a-b)))
def psnr(a,b, data_range=None):
    msev = ((a-b)**2).mean()
    if msev == 0: return float('inf')
    if data_range is None:
        data_range = a.max() - a.min()
        if data_range == 0: data_range = 1.0
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(msev)
def snr(a,b):
    signal_power = (a**2).mean()
    noise_power = ((a-b)**2).mean()
    if noise_power==0: return float('inf')
    return 10.0 * np.log10(signal_power / noise_power)

def find_best_shift_by_xcorr(ref, target, max_shift=None):
    # returns shift (target shifted by +shift aligns to ref)
    # compute xcorr, limited search window
    if max_shift is None:
        max_shift = min(len(ref), len(target))//2
    # use FFT-based xcorr on shorter arrays
    c = signal.correlate(ref - ref.mean(), target - target.mean(), mode='full')
    lag = np.arange(-len(target)+1, len(ref))
    best = np.argmax(c)
    best_lag = lag[best]
    # ensure in [-max_shift, max_shift]
    if abs(best_lag) > max_shift:
        return None
    return int(best_lag)

def crop_or_shift(ref, target, shift=None):
    # return (ref_crop, target_crop) with same length
    if shift is None:
        L = min(len(ref), len(target))
        return ref[:L], target[:L]
    # if shift >=0 means target is shifted right => target[i+shift] aligns ref[i]
    if shift >= 0:
        # overlapping region: ref[0:len(target)-shift] vs target[shift:]
        s = shift
        L = min(len(ref), len(target)-s)
        if L <= 0:
            raise ValueError("No overlap for given positive shift")
        return ref[:L], target[s:s+L]
    else:
        s = -shift
        L = min(len(ref)-s, len(target))
        if L <= 0:
            raise ValueError("No overlap for given negative shift")
        return ref[s:s+L], target[:L]

def ensure_1d(arr):
    arr = np.asarray(arr)
    if arr.ndim==2 and arr.shape[1]==1:
        return arr.squeeze(1)
    return arr.squeeze()

def main(args):
    os.makedirs(args.out, exist_ok=True)

    full_raw = np.load(args.full_raw)
    rec = np.load(args.rec)

    full_raw = ensure_1d(full_raw)
    rec = ensure_1d(rec)

    print(f"[info] raw len = {len(full_raw)}, rec len = {len(rec)}")

    shift = None
    if args.align == 'xcorr':
        print("[info] computing cross-correlation to find best alignment...")
        best = find_best_shift_by_xcorr(full_raw, rec, max_shift=args.max_shift)
        print("[info] best lag (target shifted by +lag aligns to ref):", best)
        if best is None:
            print("[warn] cross-corr found no reasonable shift; will fallback to simple crop")
        else:
            shift = best

    ref_crop, rec_crop = crop_or_shift(full_raw, rec, shift=shift)
    L = len(ref_crop)
    print(f"[info] comparison length after align/crop = {L}")

    # compute metrics
    metrics = {}
    metrics['MSE'] = mse(ref_crop, rec_crop)
    metrics['RMSE'] = rmse(ref_crop, rec_crop)
    metrics['MAE'] = mae(ref_crop, rec_crop)
    metrics['PSNR'] = psnr(ref_crop, rec_crop, data_range=(ref_crop.max()-ref_crop.min()))
    metrics['SNR_dB'] = snr(ref_crop, rec_crop)
    metrics['Pearson_r'] = float(stats.pearsonr(ref_crop, rec_crop)[0])
    metrics['length'] = int(L)
    metrics['shift_used'] = int(shift) if shift is not None else None

    # per-window MSE (optional windowed summary)
    win = args.window
    if win is None:
        win = min(32, L)
    nwin = L // win
    per_win_mse = []
    for i in range(nwin):
        a = ref_crop[i*win:(i+1)*win]
        b = rec_crop[i*win:(i+1)*win]
        per_win_mse.append(((a-b)**2).mean())
    metrics['per_window_mse_mean'] = float(np.mean(per_win_mse))
    metrics['per_window_mse_median'] = float(np.median(per_win_mse))

    # save metrics
    import json
    with open(os.path.join(args.out, "comparison_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("[info] saved metrics ->", os.path.join(args.out, "comparison_metrics.json"))
    print(metrics)

    # PLOTS
    # 1) full trace overlay (cropped)
    plt.figure(figsize=(14,3))
    plt.plot(ref_crop, label='original (raw)', linewidth=1)
    plt.plot(rec_crop, label='reconstructed', linewidth=1, alpha=0.9)
    plt.legend(); plt.title("Full sequence overlay (cropped/aligned)")
    plt.xlabel("time step"); plt.ylabel("signal")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "full_overlay.png"), dpi=300)
    plt.close()

    # 2) error time series
    err = ref_crop - rec_crop
    plt.figure(figsize=(14,2.4))
    plt.plot(err, linewidth=0.6)
    plt.title("Error (original - reconstructed) time series")
    plt.xlabel("time step"); plt.ylabel("error")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "error_time_series.png"), dpi=300)
    plt.close()

    # 3) histogram of error
    plt.figure(figsize=(5,3))
    plt.hist(err, bins=200)
    plt.title("Error histogram")
    plt.xlabel("error"); plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "error_hist.png"), dpi=300)
    plt.close()

    # 4) zoomed-in windows: plot a few worst windows by per_win_mse (top K)
    idxs = np.argsort(per_win_mse)[-args.top_k:]  # worst
    for rank, idx in enumerate(idxs[::-1], start=1):
        a = ref_crop[idx*win:(idx+1)*win]
        b = rec_crop[idx*win:(idx+1)*win]
        plt.figure(figsize=(6,2))
        plt.plot(a, label='orig'); plt.plot(b, label='rec'); plt.title(f"Worst window #{rank} idx={idx} mse={per_win_mse[idx]:.4e}")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"worst_window_{rank}_idx{idx}.png"), dpi=300)
        plt.close()

    # 5) save csv of per-window mse
    import csv
    with open(os.path.join(args.out, "per_window_mse.csv"), "w", newline='') as f:
        w = csv.writer(f); w.writerow(["window_idx","mse"]); 
        for i, m in enumerate(per_win_mse): w.writerow([i, float(m)])
    print("[info] saved per_window_mse.csv and plots to", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--full_raw", required=True, help="path to full raw npy saved by assemble_raw_from_txt (1D or (L,1))")
    p.add_argument("--rec", required=True, help="path to reconstructed sequence npy (test_rec_seq.npy)")
    p.add_argument("--out", required=True, help="output directory to save metrics/plots")
    p.add_argument("--align", choices=[None, "xcorr"], default=None, help="whether to find best shift by cross-correlation")
    p.add_argument("--max_shift", type=int, default=None, help="max shift to search (None -> half length)")
    p.add_argument("--window", type=int, default=32, help="window size for per-window MSE aggregation")
    p.add_argument("--top_k", type=int, default=6, help="how many worst windows to save plots for")
    args = p.parse_args()
    main(args)
