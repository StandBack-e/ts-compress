# scripts/run_dct.py
import numpy as np
from scipy.fftpack import dct, idct
import argparse, os, csv, time

def encode_dct(x, k):
    c = dct(x, norm='ortho')
    idx = np.argsort(np.abs(c))[::-1][:k]
    vals = c[idx]
    return idx, vals

def decode_dct(idx, vals, T):
    c = np.zeros(T)
    c[idx] = vals
    return idct(c, norm='ortho')

def bits_estimate_dct(k, T, value_bits=16):
    # index bits per coef = ceil(log2(T))
    import math
    idx_bits = math.ceil(math.log2(T))
    total_bits = k * (idx_bits + value_bits)
    return total_bits

def run(fname, out_csv, k_list=[8,16,32,64], value_bits=16):
    arr = np.load(fname)  # shape (N, T) or (N,T,1)
    if arr.ndim==3:
        arr = arr.squeeze(-1)
    N,T = arr.shape
    rows=[]
    for k in k_list:
        mses=[]
        t0 = time.time()
        bits = bits_estimate_dct(k, T, value_bits)
        for i in range(N):
            x = arr[i]
            idx, vals = encode_dct(x, k)
            xhat = decode_dct(idx, vals, T)
            mses.append(((x-xhat)**2).mean())
        t1 = time.time()
        rows.append({'method':'dct','k':k,'MSE':float(np.mean(mses)),'bits_per_window':bits,'time_s':t1-t0})
        print(rows[-1])
    # save csv
    keys = rows[0].keys()
    with open(out_csv,'w',newline='') as f:
        writer=csv.DictWriter(f,keys)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--in", dest="fin", required=True)
    p.add_argument("--out", dest="fout", default="results/dct_results.csv")
    p.add_argument("--ks", nargs="+", type=int, default=[8,16,32,64])
    args=p.parse_args()
    os.makedirs(os.path.dirname(args.fout), exist_ok=True)
    run(args.fin, args.fout, args.ks)
