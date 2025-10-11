#!/usr/bin/env python3
"""
preprocess_cmapss.py (modified)

改动说明：
- 单通道窗口现在**自动扩维**为 (N, T, 1)，兼容 CNN 输入；
- 如果使用 --detrend，会同时保存趋势参数 trend_params.json（slope/intercept），便于重构；
- 如果使用 --normalize，会保存 norm stats json（mean/scale）用于反归一；
- 保持原有参数和行为，增强了健壮性与可回溯性。
"""
import os
import argparse
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler

def load_cmapss_file(path):
    df = pd.read_csv(path, sep=r'\s+', header=None)
    return df

def assemble_unit_series(df_train, df_test):
    # 简化处理：把 train 和 test（若存在）按时间顺序 concat
    return pd.concat([df_train, df_test], axis=0, ignore_index=True)

def extract_sensor_series(df, sensor_idx):
    # sensor_idx 支持 1-based（dataset doc）或 0-based（负数也可）
    if sensor_idx < 1:
        sensor_idx = sensor_idx + 1
    col_idx = 4 + sensor_idx
    if col_idx >= df.shape[1]:
        raise ValueError(f"sensor idx {sensor_idx} too large for data with {df.shape[1]} cols")
    series = df.iloc[:, col_idx].astype(float).values
    return series

def detrend_linear_with_params(x):
    """返回 (detrended_x, slope, intercept)"""
    n = len(x)
    if n < 2:
        return x, 0.0, 0.0
    t = np.arange(n)
    A = np.vstack([t, np.ones(n)]).T
    m, c = np.linalg.lstsq(A, x, rcond=None)[0]
    detrended = x - (m * t + c)
    return detrended, float(m), float(c)

def detrend_linear(x):
    y, m, c = detrend_linear_with_params(x)
    return y

def sliding_windows(series, window, stride):
    N = len(series)
    if N < window:
        return np.zeros((0, window), dtype=series.dtype)
    num = 1 + (N - window) // stride
    # use as_strided for speed, then copy
    windows = np.lib.stride_tricks.as_strided(
        series,
        shape=(num, window),
        strides=(series.strides[0]*stride, series.strides[0]),
        writeable=False
    ).copy()
    return windows

def time_split_windows(windows, test_ratio, val_ratio):
    n = len(windows)
    test_n = int(n * test_ratio)
    val_n = int(n * val_ratio)
    train_n = n - val_n - test_n
    if train_n <= 0:
        raise ValueError("train set is empty; reduce test/val ratio or increase data")
    train = windows[:train_n]
    val = windows[train_n:train_n+val_n]
    test = windows[train_n+val_n:]
    return train, val, test

def save_arrays(out_dir, prefix, train, val, test, scaler=None, trend_params=None, labels_train=None, labels_val=None, labels_test=None):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{prefix}_train.npy"), train)
    np.save(os.path.join(out_dir, f"{prefix}_val.npy"), val)
    np.save(os.path.join(out_dir, f"{prefix}_test.npy"), test)
    if labels_train is not None:
        np.save(os.path.join(out_dir, f"{prefix}_train_labels.npy"), labels_train)
        np.save(os.path.join(out_dir, f"{prefix}_val_labels.npy"), labels_val)
        np.save(os.path.join(out_dir, f"{prefix}_test_labels.npy"), labels_test)
    if scaler is not None:
        # scaler.mean_ / scale_ are arrays; we saved fitting on flattened values (1-d)
        stats = {'mean': float(scaler.mean_[0]), 'scale': float(scaler.scale_[0])}
        with open(os.path.join(out_dir, f"{prefix}_norm_stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)
    if trend_params is not None:
        with open(os.path.join(out_dir, f"{prefix}_trend_params.json"), 'w') as f:
            json.dump(trend_params, f, indent=2)
    print(f"Saved arrays to {out_dir}: train {train.shape}, val {val.shape}, test {test.shape}")

def simple_anomaly_label(window, z_thresh=6.0):
    mu = window.mean()
    sigma = window.std(ddof=0) if window.std(ddof=0) > 0 else 1.0
    z = np.abs((window - mu) / sigma)
    return int(np.any(z > z_thresh))

def main(args):
    # paths like train_FD001.txt
    train_path = os.path.join(args.input_dir, f"train_{args.subset}.txt")
    test_path = os.path.join(args.input_dir, f"test_{args.subset}.txt")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} not found. 请把 CMAPSS 数据放在 {args.input_dir}，文件名如 train_{args.subset}.txt")
    df_train = load_cmapss_file(train_path)
    if os.path.exists(test_path):
        df_test = load_cmapss_file(test_path)
    else:
        df_test = pd.DataFrame(columns=df_train.columns)  # empty

    df_all = assemble_unit_series(df_train, df_test)
    series = extract_sensor_series(df_all, args.sensor)

    trend_params = None
    # optional detrend (and save slope/intercept)
    if args.detrend:
        full_series_length = len(series) # 保存去趋势前的完整长度
        series, slope, intercept = detrend_linear_with_params(series)
        trend_params = {'slope': slope, 'intercept': intercept, 'full_series_length': full_series_length}
        print(f"[preprocess] detrend applied: slope={slope:.6e}, intercept={intercept:.6e}")

    # sliding windows
    windows = sliding_windows(series.astype(np.float32), args.window, args.stride)
    if windows.shape[0] == 0:
        raise RuntimeError("No windows generated: check window/stride/series length")

    # optional normalization: fit scaler on train portion (time-aware)
    split_idx = int(np.ceil(len(windows) * (1 - args.val_ratio - args.test_ratio)))
    scaler = None
    if args.normalize:
        train_for_stats = windows[:split_idx].reshape(-1, 1)
        scaler = StandardScaler()
        scaler.fit(train_for_stats)
        def apply_scaler(ws):
            s = ws.copy().reshape(-1,1)
            s = scaler.transform(s).reshape(ws.shape)
            return s
        windows = apply_scaler(windows)
        print(f"[preprocess] normalization applied (fit on first {split_idx} windows)")

    # simple labels BEFORE normalization (if requested)
    labels = None
    if args.make_labels:
        labels = np.array([simple_anomaly_label(w) for w in windows], dtype=np.int64)
        print(f"[preprocess] generated labels (n={len(labels)})")

    # split by time
    train, val, test = time_split_windows(windows, args.test_ratio, args.val_ratio)
    if labels is not None:
        labels_train, labels_val, labels_test = time_split_windows(labels, args.test_ratio, args.val_ratio)
    else:
        labels_train = labels_val = labels_test = None

    # automatic expand single-channel -> (N, T, 1)
    # if already multi-channel (3D), keep as is
    if train.ndim == 2:
        train = train[..., None]
    if val.ndim == 2:
        val = val[..., None]
    if test.ndim == 2:
        test = test[..., None]

    save_arrays(args.out_dir, args.save_prefix, train, val, test, scaler=scaler, trend_params=trend_params,
                labels_train=labels_train, labels_val=labels_val, labels_test=labels_test)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subset", type=str, default="FD001", help="FD001/FD002/FD003/FD004")
    p.add_argument("--sensor", type=int, default=2, help="sensor index (1-based within 1..21) or use 0-based negative")
    p.add_argument("--input_dir", type=str, default="./data/CMAPSS", help="dir containing CMAPSS txt files")
    p.add_argument("--out_dir", type=str, default="./data/prepared", help="output dir")
    p.add_argument("--window", type=int, default=128, help="window length in samples")
    p.add_argument("--stride", type=int, default=64, help="stride in samples")
    p.add_argument("--detrend", action="store_true", help="apply linear detrend and save trend params")
    p.add_argument("--normalize", action="store_true", help="apply z-score normalization (fit on train portion) and save stats")
    p.add_argument("--make_labels", action="store_true", help="generate simple anomaly labels based on z-threshold")
    p.add_argument("--test_ratio", type=float, default=0.15, help="test ratio (time split)")
    p.add_argument("--val_ratio", type=float, default=0.15, help="val ratio (time split)")
    p.add_argument("--save_prefix", type=str, default="cmapss", help="file prefix for saved arrays")
    args = p.parse_args()
    main(args)
