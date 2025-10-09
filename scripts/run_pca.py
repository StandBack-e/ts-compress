# scripts/run_pca.py
import numpy as np, time, os, csv
from sklearn.decomposition import PCA
import argparse

def bits_estimate_pca(n_comp, T, value_bits=16):
    # coefficients: n_comp per window, plus mean vector (T) amortized
    import math
    coeff_bits = n_comp * value_bits
    # amortize mean vector cost over many windows => ignored here or amortize externally
    return coeff_bits

def run(fin, fout, n_components=8, value_bits=16):
    arr = np.load(fin)
    if arr.ndim==3:
        arr = arr.squeeze(-1)  # (N,T)
    N, T = arr.shape
    print("PCA run:", fin, "N,T=", N, T)
    t0=time.time()
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(arr)  # (N, n_components)
    Xrec = pca.inverse_transform(Z)
    t1=time.time()
    mse = float(((arr - Xrec)**2).mean())
    bits = bits_estimate_pca(n_components, T, value_bits)
    row = {'method':'pca','n_components':n_components,'MSE':mse,'bits_per_window':bits,'time_s':t1-t0}
    os.makedirs(os.path.dirname(fout), exist_ok=True)
    with open(fout,'w',newline='') as f:
        writer=csv.DictWriter(f,row.keys())
        writer.writeheader(); writer.writerow(row)
    print("done", row)

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="fin", required=True)
    p.add_argument("--out", dest="fout", default="results/pca.csv")
    p.add_argument("--n", dest="ncomp", type=int, default=8)
    p.add_argument("--value_bits", type=int, default=16)
    args = p.parse_args()
    run(args.fin, args.fout, args.ncomp, args.value_bits)
