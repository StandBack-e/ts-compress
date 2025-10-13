# scripts/assemble_raw_from_txt.py
"""
从 CMAPSS 原始 txt 拼回指定 sensor 的完整原始序列并保存为 npy。
用法示例（powershell）:
python scripts/assemble_raw_from_txt.py --subset FD001 --sensor 2 --input_dir ./data/CMAPSS --out_dir ./data/prepared/FD001_w32_s16_norm_detrend --save_prefix cmapss_w32_s16_norm_detrend
"""
import os
import argparse
import numpy as np
import pandas as pd

def load_cmapss_file(path):
    df = pd.read_csv(path, sep=r'\s+', header=None)
    return df

def extract_sensor_series_from_files(input_dir, subset, sensor_idx):
    train_path = os.path.join(input_dir, f"train_{subset}.txt")
    test_path = os.path.join(input_dir, f"test_{subset}.txt")
    if not os.path.exists(train_path):
        raise FileNotFoundError(train_path)
    df_train = load_cmapss_file(train_path)
    if os.path.exists(test_path):
        df_test = load_cmapss_file(test_path)
        df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    else:
        df_all = df_train
    if sensor_idx < 1:
        sensor_idx = sensor_idx + 1
    col_idx = 4 + sensor_idx
    if col_idx >= df_all.shape[1]:
        raise ValueError(f"sensor_idx {sensor_idx} too large; df has {df_all.shape[1]} cols")
    series = df_all.iloc[:, col_idx].astype(float).values
    return series

def save_full_raw(series, out_dir, save_prefix):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{save_prefix}_full_raw.npy")
    np.save(out_path, series)
    print("Saved full raw sequence to:", out_path)
    return out_path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subset", type=str, default="FD001")
    p.add_argument("--sensor", type=int, default=2)
    p.add_argument("--input_dir", type=str, default="./data/CMAPSS")
    p.add_argument("--out_dir", type=str, required=True, help="prepared dir where you put preprocessed npys")
    p.add_argument("--save_prefix", type=str, default="cmapss")
    args = p.parse_args()

    series = extract_sensor_series_from_files(args.input_dir, args.subset, args.sensor)
    save_full_raw(series, args.out_dir, args.save_prefix)
