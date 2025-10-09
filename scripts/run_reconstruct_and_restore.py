# scripts/run_reconstruct_and_restore.py
import numpy as np, json, os
from reconstruct import reconstruct_from_windows
from reconstruct import inverse_normalize, add_linear_trend, estimate_and_add_trend

# 参数（根据你的 preprocess）
WINDOW = 32    # T
STRIDE = 16    # S
PREPARED_DIR = "data/prepared"
OUT_DIR = "results/reconstructed"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) load windows (example: use test windows recon from your eval)
windows = np.load("results/sample_rec.npy")  # (N_save, T, C) or (N, T, C)
# if you want reconstruct full prepared train set instead:
# windows = np.load(os.path.join(PREPARED_DIR, "train.npy"))

# 2) if windows are only a subset (sample_rec) then you won't reconstruct whole original.
#    Normally run this on full windows arrays
windows_full = np.load(os.path.join(PREPARED_DIR, "train.npy"))  # use train/val/test as needed

# Example: reconstruct train set back to a long series
seq, coverage = reconstruct_from_windows(windows_full, WINDOW, STRIDE)  # seq shape (L, C) or (L,)

# 3) inverse normalization if stats exist
stats_path = os.path.join(PREPARED_DIR, "cmapss_norm_stats.json")  # or 'cmapss_norm_stats.json' depending on your save_prefix
if os.path.exists(stats_path):
    seq = inverse_normalize(seq, stats_path)
else:
    print("no norm stats found; skipping inverse normalize")

# 4) inverse detrend: if you saved slope/intercept use add_linear_trend; else estimate
trend_params_path = os.path.join(PREPARED_DIR, "trend_params.json")
if os.path.exists(trend_params_path):
    p = json.load(open(trend_params_path))
    seq = add_linear_trend(seq, p['slope'], p['intercept'])
else:
    print("trend params not found; estimating trend and adding back (approx)")
    seq = estimate_and_add_trend(seq)

# 5) save reconstructed sequence
np.save(os.path.join(OUT_DIR, "reconstructed_train_seq.npy"), seq)
np.save(os.path.join(OUT_DIR, "coverage.npy"), coverage)
print("Saved reconstructed sequence to", OUT_DIR)
