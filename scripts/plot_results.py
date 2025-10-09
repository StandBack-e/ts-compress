# scripts/plot_results.py
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_waveforms(orig_path, rec_path, out_png="results/recon_examples.png", n=6):
    orig = np.load(orig_path)  # (N,T,C)
    rec  = np.load(rec_path)
    N = min(n, orig.shape[0])
    T = orig.shape[1]
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig, axs = plt.subplots(N, 1, figsize=(8, 1.6*N), sharex=True)
    if N == 1:
        axs = [axs]
    for i in range(N):
        axs[i].plot(orig[i].squeeze(), label='orig', linewidth=1)
        axs[i].plot(rec[i].squeeze(), label='rec', linewidth=1, alpha=0.9)
        axs[i].legend(loc='upper right', fontsize=8)
        axs[i].set_ylabel(f"sample {i}")
    axs[-1].set_xlabel("time step")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print("Saved waveform figure:", out_png)
    plt.close()

def plot_per_window_mse(csv_path, out_png="results/per_window_mse.png"):
    df = pd.read_csv(csv_path)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(10,3))
    plt.plot(df['index'], df['mse'], marker='.', markersize=3, linewidth=0.5)
    plt.yscale('log')
    plt.xlabel("window index")
    plt.ylabel("MSE (log scale)")
    plt.title("Per-window reconstruction MSE")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print("Saved per-window mse:", out_png)
    plt.close()

def plot_mse_hist(csv_path, out_png="results/mse_hist.png"):
    df = pd.read_csv(csv_path)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.hist(df['mse'], bins=50)
    plt.xlabel("MSE")
    plt.ylabel("count")
    plt.title("Distribution of per-window MSE")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print("Saved mse hist:", out_png)
    plt.close()

def plot_pareto(results_csv_list, labels, out_png="results/pareto.png"):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(6,4))
    for f, lbl in zip(results_csv_list, labels):
        df = pd.read_csv(f)
        plt.scatter(df['bits_per_window'], df['MSE'], label=lbl)
        for _, r in df.iterrows():
            label_txt = str(int(r.get('k', r.get('K', 0)))) if ('k' in r or 'K' in r) else ''
            plt.text(r['bits_per_window'], r['MSE'], label_txt, fontsize=8)
    plt.xscale('log')
    plt.xlabel('bits per window')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print("Saved pareto:", out_png)
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig", default="results/sample_orig.npy")
    parser.add_argument("--rec", default="results/sample_rec.npy")
    parser.add_argument("--mse_csv", default="results/test_per_window_mse.csv")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()
    plot_waveforms(args.orig, args.rec, out_png=os.path.join(args.out_dir,"recon_examples.png"), n=8)
    plot_per_window_mse(args.mse_csv, out_png=os.path.join(args.out_dir,"per_window_mse.png"))
    plot_mse_hist(args.mse_csv, out_png=os.path.join(args.out_dir,"mse_hist.png"))
