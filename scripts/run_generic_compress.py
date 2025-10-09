# scripts/run_generic_compress.py
import numpy as np, os, time, csv, argparse, io
import gzip, brotli
from pathlib import Path

def quantize_array(arr, value_bits=16):
    # uniform quantize to signed integer range
    # arr expected float in reasonable range; we map to int16 or smaller
    # here we map by min/max of arr to signed range
    assert value_bits in (8,16)
    if value_bits==8:
        qmin, qmax = -128, 127
        dtype = np.int8
    else:
        qmin, qmax = -32768, 32767
        dtype = np.int16
    amin = arr.min(); amax = arr.max()
    if amin==amax:
        return np.zeros_like(arr, dtype=dtype)
    # linear map
    scaled = (arr - amin) / (amax - amin)  # [0,1]
    ints = (scaled * (qmax - qmin) + qmin).round().astype(dtype)
    return ints, float(amin), float(amax)

def array_to_bytes(ints):
    return ints.tobytes()

def compress_bytes_gzip(b):
    return gzip.compress(b)

def compress_bytes_brotli(b, quality=5):
    return brotli.compress(b, quality=quality)

def run(fin, fout, preprocess_list=['raw','delta'], compressors=['gzip','brotli'], value_bits=16):
    arr = np.load(fin)
    if arr.ndim==3:
        arr = arr.squeeze(-1)
    N,T = arr.shape
    rows=[]
    for prep in preprocess_list:
        # prepare data per window (we will compress the whole concatenated stream to estimate average)
        if prep=='raw':
            proc = arr.copy()
        elif prep=='delta':
            # first-order diff per window: keep first sample then diffs
            proc = np.concatenate([arr[:, :1], np.diff(arr, axis=1)], axis=1)
        elif prep=='delta_global':
            # global difference across windows (not usual) - skip
            proc = np.concatenate([arr[:, :1], np.diff(arr, axis=1)], axis=1)
        else:
            raise ValueError("unsupported prep "+prep)

        # quantize per-window or global? We'll quantize globally for fairness
        flat = proc.flatten()
        ints, amin, amax = quantize_array(flat, value_bits=value_bits)
        # reconstruct ints as original windowed shape
        ints = ints.reshape(proc.shape)
        # concatenate all window bytes
        b = ints.tobytes()
        for comp in compressors:
            t0=time.time()
            if comp=='gzip':
                out = compress_bytes_gzip(b)
            elif comp=='brotli':
                out = compress_bytes_brotli(b)
            else:
                raise ValueError("unsupported compressor")
            t1=time.time()
            compressed_bytes = len(out)
            bits_per_window = compressed_bytes * 8.0 / N
            row = {'method': comp, 'prep': prep, 'value_bits': value_bits,
                   'MSE': float(((arr - proc)**2).mean()), # note: MSE between raw and proc (delta differs) -- for fairness we should reconstruct; keep as proxy
                   'bits_per_window': bits_per_window, 'time_s': t1-t0, 'amin':amin, 'amax':amax}
            rows.append(row)
            print(row)
    # save CSV
    os.makedirs(os.path.dirname(fout), exist_ok=True)
    keys = rows[0].keys()
    with open(fout,'w',newline='') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader(); writer.writerows(rows)
    print("saved ->", fout)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument("--in", dest="fin", required=True)
    p.add_argument("--out", dest="fout", default="results/generic_compress.csv")
    p.add_argument("--value_bits", type=int, default=16)
    p.add_argument("--compressors", nargs="+", default=['gzip','brotli'])
    p.add_argument("--preps", nargs="+", default=['raw','delta'])
    args=p.parse_args()
    run(args.fin, args.fout, preprocess_list=args.preps, compressors=args.compressors, value_bits=args.value_bits)
