# eval_saved.py
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
from torch.utils.data import DataLoader
from data.loaders import TimeSeriesDataset
from models.cnn_ae import CNNAutoEncoder
from models.cnn_rnn_ae import CNNRNNAutoEncoder

def reconstruct_from_windows_np(windows, window, stride):
    if windows.ndim == 2:
        windows = windows[:, :, None]
    N, T, C = windows.shape
    assert T == window, f"window mismatch T={T} vs window arg {window}"
    L = (N - 1) * stride + window
    seq = np.zeros((L, C), dtype=windows.dtype)
    coverage = np.zeros((L,), dtype=np.int32)
    for i in range(N):
        s = i * stride
        e = s + window
        seq[s:e] += windows[i]
        coverage[s:e] += 1
    mask = coverage > 0
    seq[mask] = seq[mask] / coverage[mask, None]
    return seq.squeeze(), coverage

def inverse_normalize_np(seq, norm_stats_path):
    if not norm_stats_path or not os.path.exists(norm_stats_path):
        return seq
    stats = json.load(open(norm_stats_path))
    mean = float(stats['mean'])
    scale = float(stats['scale'])
    return seq * scale + mean

def add_linear_trend_np(seq, slope, intercept):
    L = seq.shape[0]
    t = np.arange(L)
    trend = slope * t + intercept
    if seq.ndim == 2:
        return seq + trend[:, None]
    return seq + trend

def estimate_and_add_trend_np(seq):
    L = seq.shape[0]
    t = np.arange(L)
    if seq.ndim == 2:
        out = seq.copy()
        for c in range(seq.shape[1]):
            y = seq[:, c]
            A = np.vstack([t, np.ones(L)]).T
            m, b = np.linalg.lstsq(A, y, rcond=None)[0]
            out[:, c] = out[:, c] + (m * t + b)
        return out
    else:
        y = seq
        A = np.vstack([t, np.ones(L)]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return seq + (m * t + b)

def find_json_file_with_keyword(dirpath, keyword):
    if not os.path.isdir(dirpath):
        return None
    for fname in os.listdir(dirpath):
        if fname.lower().endswith('.json') and keyword.lower() in fname.lower():
            return os.path.join(dirpath, fname)
    return None

def infer_stride_from_dirname(prepared_dir):
    basename = os.path.basename(prepared_dir)
    m = re.search(r'w(\d+)_s(\d+)', basename)
    if m:
        return int(m.group(2))
    m2 = re.search(r'_s(\d+)', basename)
    if m2:
        return int(m2.group(1))
    return None

def safe_filesize(path):
    try:
        return os.path.getsize(path)
    except Exception:
        return None

def compute_cr(raw_bytes, comp_bytes):
    """返回 compression ratio (raw_bytes/comp_bytes) 和 percent savings (0-1)"""
    if raw_bytes is None or comp_bytes is None or comp_bytes == 0:
        return None, None
    cr = float(raw_bytes) / float(comp_bytes)
    saving = 1.0 - (float(comp_bytes) / float(raw_bytes))
    return cr, saving

def evaluate_and_save(cfg, n_save=8, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = torch.load(cfg['train']['ckpt'], map_location='cpu')
    test_arr = np.load(cfg['data']['test_np'])
    ds = TimeSeriesDataset(test_arr)
    dl = DataLoader(ds, batch_size=cfg['train']['batch_size'], shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T, C = test_arr.shape[1], test_arr.shape[2]
    if cfg['model']['type'] == 'cnn':
        model = CNNAutoEncoder(T=T, C=C, latent_dim=cfg['model']['latent_dim']).to(device)
    else:
        model = CNNRNNAutoEncoder(T=T, C=C, latent_dim=cfg['model']['latent_dim']).to(device)
    # model = CNNAutoEncoder(T=T, C=C, latent_dim=cfg['model']['latent_dim']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    mse_list = []
    all_orig = []
    all_rec  = []
    all_latent = []

    with torch.no_grad():
        for xb in dl:
            xb = xb.to(device).float()
            z = model(xb)
            rec = model.decode(z)
            # rec, z = model(xb)
            b_mse = ((xb - rec)**2).mean(dim=(1,2)).cpu().numpy()  # (B,)
            mse_list.extend(b_mse.tolist())
            all_orig.append(xb.cpu().numpy())
            all_rec.append(rec.cpu().numpy())
            # ensure latent is 2D (N, latent_dim)
            all_latent.append(z.cpu().numpy())

    all_orig = np.concatenate(all_orig, axis=0)  # (N, T, C)
    all_rec  = np.concatenate(all_rec, axis=0)
    all_latent = np.concatenate(all_latent, axis=0)  # (N, latent_dim)
    overall_mse = float(np.mean(mse_list))
    print("Test overall MSE:", overall_mse)

    # Save window-level arrays
    np.save(os.path.join(out_dir, "test_orig_windows.npy"), all_orig)
    np.save(os.path.join(out_dir, "test_rec_windows.npy"), all_rec)

    # Save latents (raw float32)
    latent_path = os.path.join(out_dir, "test_latents.npy")
    np.save(latent_path, all_latent.astype(np.float32))
    print("Saved latent features ->", latent_path)
    # Save compressed latent (npz) to simulate compressed bytes
    latent_npz = os.path.join(out_dir, "test_latents.npz")
    np.savez_compressed(latent_npz, all_latent.astype(np.float32))
    print("Saved compressed latent (npz) ->", latent_npz)

    # save per-window mse csv
    csv_path = os.path.join(out_dir, "test_per_window_mse.csv")
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["index","mse"])
        for i,m in enumerate(mse_list):
            writer.writerow([i, m])
    print("Saved per-window mse ->", csv_path)

    # get stride: priority cfg['data']['stride'] > infer from prepared dir name > default 1 (warn)
    prepared_dir = os.path.dirname(cfg['data']['test_np'])
    stride = cfg.get('data', {}).get('stride', None)
    if stride is None:
        stride = infer_stride_from_dirname(prepared_dir)
    if stride is None:
        print("[eval_saved] WARNING: stride not found in cfg or inferred; defaulting to 1 (likely wrong). Set cfg['data']['stride'].")
        stride = 1
    else:
        print(f"[eval_saved] using stride = {stride}")

    # Reconstruct full sequences from windows (still normalized/detrended if that was applied)
    recon_seq_normed, coverage = reconstruct_from_windows_np(all_rec, T, stride)
    orig_seq_normed, _ = reconstruct_from_windows_np(all_orig, T, stride)

    print("[eval_saved] Saving sequences in their processed (normalized) state.")
    np.save(os.path.join(out_dir, "test_rec_seq.npy"), recon_seq_normed)
    np.save(os.path.join(out_dir, "test_orig_seq.npy"), orig_seq_normed)

    recon_seq_restored = recon_seq_normed
    orig_seq_restored = orig_seq_normed


    # detect norm/trend files in prepared dir
    norm_stats_path = find_json_file_with_keyword(prepared_dir, 'norm_stats')
    trend_params_path = find_json_file_with_keyword(prepared_dir, 'trend_params')

    # inverse normalize
    if norm_stats_path:
        recon_seq_restored = inverse_normalize_np(recon_seq_normed, norm_stats_path)
        orig_seq_restored = inverse_normalize_np(orig_seq_normed, norm_stats_path)
        print(f"[eval_saved] applied inverse normalize from: {norm_stats_path}")
    else:
        recon_seq_restored = recon_seq_normed
        orig_seq_restored = orig_seq_normed
        print("[eval_saved] no norm_stats found; kept normalized values")

    # inverse detrend
    if trend_params_path:
        tp = json.load(open(trend_params_path))
        recon_seq_restored = add_linear_trend_np(recon_seq_restored, tp.get('slope',0.0), tp.get('intercept',0.0))
        orig_seq_restored = add_linear_trend_np(orig_seq_restored, tp.get('slope',0.0), tp.get('intercept',0.0))
        print(f"[eval_saved] applied trend add from: {trend_params_path}")
    else:
        # recon_seq = estimate_and_add_trend_np(recon_seq)
        # orig_seq = estimate_and_add_trend_np(orig_seq)
        print("[eval_saved] no trend params found; did estimate_and_add_trend (approx)")

    # save full sequence arrays and coverage
    # np.save(os.path.join(out_dir, "test_rec_seq.npy"), recon_seq)
    # np.save(os.path.join(out_dir, "test_orig_seq.npy"), orig_seq)
    np.save(os.path.join(out_dir, "coverage.npy"), coverage)

    # collect file size info for summary
    latent_bytes = safe_filesize(latent_path)
    latent_npz_bytes = safe_filesize(latent_npz)

    # try to find full_raw to report size (optional)
    full_raw_path = None
    for fname in os.listdir(prepared_dir):
        if 'full_raw' in fname and fname.endswith('.npy'):
            full_raw_path = os.path.join(prepared_dir, fname)
            print(f"[eval_saved] found full_raw: {full_raw_path}")
            break
    full_raw_bytes = safe_filesize(full_raw_path) if full_raw_path is not None else None

    # compute compression ratios (raw npy vs latent npy and vs latent npz)
    cr_raw_vs_latent, saving_raw_vs_latent = compute_cr(full_raw_bytes, latent_bytes) if full_raw_bytes else (None, None)
    cr_raw_vs_npz, saving_raw_vs_npz = compute_cr(full_raw_bytes, latent_npz_bytes) if full_raw_bytes else (None, None)

    # save summary (add latent sizes and CR)
    summary = {
        "overall_mse": overall_mse,
        "n_windows": len(mse_list),
        "T": T,
        "C": C,
        "prepared_dir": prepared_dir,
        "norm_stats_path": norm_stats_path,
        "trend_params_path": trend_params_path,
        "stride": stride,
        "latent_bytes": latent_bytes,
        "latent_compressed_bytes": latent_npz_bytes,
        "full_raw_path": full_raw_path,
        "full_raw_bytes": full_raw_bytes,
        "compression_ratio_raw_vs_latent_npy": cr_raw_vs_latent,
        "compression_saving_raw_vs_latent_npy": saving_raw_vs_latent,
        "compression_ratio_raw_vs_latent_npz": cr_raw_vs_npz,
        "compression_saving_raw_vs_latent_npz": saving_raw_vs_npz
    }
    with open(os.path.join(out_dir,"summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # plot full sequence comparison (crop to same length)
    L = min(len(orig_seq_restored), len(recon_seq_restored))
    plt.figure(figsize=(12,3))
    if orig_seq_restored.ndim == 2:
        plt.plot(orig_seq_restored[:L,0], label='orig')
        plt.plot(recon_seq_restored[:L,0], label='recon', alpha=0.9)
    else:
        plt.plot(orig_seq_restored[:L], label='orig')
        plt.plot(recon_seq_restored[:L], label='recon', alpha=0.9)
    plt.legend(); plt.title("Full-sequence reconstruction (from windows)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "recon_full_sequence.png"), dpi=300)
    plt.close()

    print("Saved windows, full-sequence recon, coverage, latents and plots to", out_dir)
    return overall_mse, os.path.abspath(out_dir)

if __name__ == "__main__":
    import yaml, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/config.yaml")
    parser.add_argument("--n_save", type=int, default=8, help="how many example windows to save")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg,encoding='utf-8'))
    evaluate_and_save(cfg, n_save=args.n_save, out_dir=args.out_dir)
