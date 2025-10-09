#!/usr/bin/env python3
"""
gen_synthetic.py

功能：
- 生成合成1维时序数据：低频基带 + 高频噪声 + 随机尖峰/跳变用作异常
- 支持按秒数和采样率定义长度，支持多段不同模式合成
- 可直接切片成 windows 并保存 train/val/test npy 和 labels
- 用于做可控消融与可视化示例

用法示例：
python data/gen_synthetic.py --out_dir ./data/synth --duration 600 --fs 20 --window 128 --stride 64 --anomaly_rate 0.002 --seed 42
"""

import os
import argparse
import numpy as np
import json

def generate_signal(duration_s=600, fs=20, C=1, rng=None,
                    base_freqs=None, noise_std=0.05,
                    spike_prob=0.0005, spike_ampl_scale=5.0,
                    drift=False):
    N = int(duration_s * fs)
    t = np.arange(N) / fs
    if rng is None:
        rng = np.random.RandomState(0)
    if base_freqs is None:
        base_freqs = [0.05]
    sig = np.zeros((N, C), dtype=np.float32)
    for c in range(C):
        base = np.zeros(N)
        for f in base_freqs:
            amp = 1.0 + 0.3 * c
            base += amp * np.sin(2 * np.pi * f * t + rng.rand() * 2*np.pi)
        noise = noise_std * rng.randn(N)
        sig[:, c] = base + noise
        if drift:
            sig[:, c] += 0.001 * t  # small linear drift
    # add spikes (anomalies)
    if spike_prob > 0:
        mask = rng.rand(N, C) < spike_prob
        spikes = (rng.randn(N, C) * spike_ampl_scale) * mask
        sig += spikes
    return sig.squeeze()  # (N,) if C==1 else (N,C)

def sliding_windows(arr, window, stride):
    N = len(arr)
    if N < window:
        return np.zeros((0, window))
    num = 1 + (N - window) // stride
    return np.lib.stride_tricks.as_strided(
        arr,
        shape=(num, window),
        strides=(arr.strides[0]*stride, arr.strides[0]),
        writeable=False
    ).copy()

def make_labels_from_spikes(arr, window, stride, z_thresh=6.0):
    # simple labeling: window labeled 1 if any point magnitude exceeds mean+z_thresh*std
    w = sliding_windows(arr, window, stride)
    labels = []
    for win in w:
        mu = win.mean()
        sigma = win.std(ddof=0) if win.std(ddof=0) > 0 else 1.0
        if np.any(np.abs((win - mu)/sigma) > z_thresh):
            labels.append(1)
        else:
            labels.append(0)
    return np.array(labels, dtype=np.int64)

def save_windows(out_dir, prefix, windows, labels=None):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{prefix}_windows.npy"), windows)
    if labels is not None:
        np.save(os.path.join(out_dir, f"{prefix}_labels.npy"), labels)
    print("Saved:", os.path.join(out_dir, f"{prefix}_windows.npy"), "shape:", windows.shape)

def time_split(windows, test_ratio, val_ratio):
    n = len(windows)
    test_n = int(n * test_ratio)
    val_n = int(n * val_ratio)
    train_n = n - val_n - test_n
    if train_n <= 0:
        raise ValueError("Not enough windows for split")
    train = windows[:train_n]
    val = windows[train_n:train_n+val_n]
    test = windows[train_n+val_n:]
    return train, val, test

def main(args):
    rng = np.random.RandomState(args.seed)
    sig = generate_signal(duration_s=args.duration, fs=args.fs, C=1, rng=rng,
                          base_freqs=[args.base_freq], noise_std=args.noise_std,
                          spike_prob=args.anomaly_rate, spike_ampl_scale=args.spike_scale,
                          drift=args.drift)
    # sliding windows
    windows = sliding_windows(sig, args.window, args.stride)
    labels = make_labels_from_spikes(sig, args.window, args.stride, z_thresh=args.z_thresh) if args.make_labels else None

    train_w, val_w, test_w = time_split(windows, args.test_ratio, args.val_ratio)
    if labels is not None:
        lab_train, lab_val, lab_test = time_split(labels, args.test_ratio, args.val_ratio)
    else:
        lab_train = lab_val = lab_test = None

    # save arrays
    out_dir = args.out_dir
    save_windows(out_dir, args.save_prefix + "_train", train_w, labels=None)
    save_windows(out_dir, args.save_prefix + "_val", val_w, labels=None)
    save_windows(out_dir, args.save_prefix + "_test", test_w, labels=None)
    if args.make_labels:
        np.save(os.path.join(out_dir, f"{args.save_prefix}_train_labels.npy"), lab_train)
        np.save(os.path.join(out_dir, f"{args.save_prefix}_val_labels.npy"), lab_val)
        np.save(os.path.join(out_dir, f"{args.save_prefix}_test_labels.npy"), lab_test)
    # also save raw signal for inspection
    np.save(os.path.join(out_dir, f"{args.save_prefix}_raw_signal.npy"), sig)
    meta = {
        "duration_s": args.duration,
        "fs": args.fs,
        "N": int(len(sig)),
        "window": args.window,
        "stride": args.stride,
        "noise_std": args.noise_std,
        "anomaly_rate": args.anomaly_rate,
        "spike_scale": args.spike_scale
    }
    with open(os.path.join(out_dir, f"{args.save_prefix}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Done. meta:", meta)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="./data/synth", help="output dir")
    p.add_argument("--duration", type=float, default=600.0, help="seconds")
    p.add_argument("--fs", type=float, default=20.0, help="sampling rate")
    p.add_argument("--window", type=int, default=128, help="window length in samples")
    p.add_argument("--stride", type=int, default=64, help="stride in samples")
    p.add_argument("--anomaly_rate", type=float, default=0.0005, help="probability per sample to generate spike")
    p.add_argument("--spike_scale", type=float, default=6.0, help="spike amplitude multiplier")
    p.add_argument("--noise_std", type=float, default=0.05, help="high-frequency noise std")
    p.add_argument("--base_freq", type=float, default=0.05, help="base sine frequency")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--make_labels", action="store_true", help="generate window-level labels")
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--save_prefix", type=str, default="synth")
    p.add_argument("--detrend", action="store_true")
    p.add_argument("--z_thresh", type=float, default=6.0)
    args = p.parse_args()
    main(args)
