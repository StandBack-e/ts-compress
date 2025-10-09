# scripts/run_ksvd.py
import numpy as np
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit
import argparse, os, csv, time, math

def estimate_bits(K, s, T, value_bits=8, amortize_N=10000):
    idx_bits = math.ceil(math.log2(K))
    bits_per_window = s * (idx_bits + value_bits) + (K*T*32)/amortize_N  # dict cost amortized (float32)
    return bits_per_window

def run(in_file, out_csv, K=128, s=8, value_bits=8, amortize_N=10000):
    arr = np.load(in_file)
    if arr.ndim==3: arr = arr.squeeze(-1)
    N,T = arr.shape
    print("training dictionary K=",K)
    dl = DictionaryLearning(n_components=K, transform_algorithm='omp', transform_n_nonzero_coefs=s, verbose=1)
    dl.fit(arr)  # might be slow
    D = dl.components_   # shape (K, T)
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=s)
    mses=[]
    t0=time.time()
    for i in range(N):
        x = arr[i]
        # dl.transform gives codes
        alpha = dl.transform(x.reshape(1,-1))[0]  # shape (K,)
        recon = alpha.dot(D)
        mses.append(((x-recon)**2).mean())
    t1=time.time()
    bits = estimate_bits(K, s, T, value_bits, amortize_N)
    row = {'method':'ksvd','K':K,'s':s,'MSE':float(np.mean(mses)),'bits_per_window':bits,'time_s':t1-t0}
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv,'w',newline='') as f:
        import csv
        writer=csv.DictWriter(f,row.keys())
        writer.writeheader(); writer.writerow(row)
    print("done", row)

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--in", dest="fin", required=True)
    p.add_argument("--out", dest="fout", default="results/ksvd.csv")
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--s", type=int, default=8)
    p.add_argument("--value_bits", type=int, default=8)
    p.add_argument("--amortize_N", type=int, default=10000)
    args=p.parse_args()
    run(args.fin, args.fout, args.K, args.s, args.value_bits, args.amortize_N)
