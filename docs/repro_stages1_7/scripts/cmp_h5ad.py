"""Compare reproduced QC h5ad files to the canonical ones: md5, then content (X, obs, var, raw) if md5 differs."""
import sys, hashlib, csv
from pathlib import Path
from multiprocessing import Pool
import numpy as np, scipy.sparse as sp, anndata as ad, pandas as pd

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()

def same_mat(a, b):
    if a is None or b is None: return a is None and b is None
    if a.shape != b.shape: return False
    a = sp.csr_matrix(a); b = sp.csr_matrix(b)
    if a.dtype != b.dtype: return f"dtype {a.dtype} vs {b.dtype} values_equal={(a != b).nnz == 0}"
    return (a != b).nnz == 0

def cmp(args):
    name, A, B = args
    pa, pb = Path(A) / name, Path(B) / name
    ma, mb = md5(pa), md5(pb)
    if ma == mb: return [name, ma, mb, 'IDENTICAL', '']
    a, b = ad.read_h5ad(pa), ad.read_h5ad(pb)
    d = []
    x = same_mat(a.X, b.X); d.append(f"X={x}")
    d.append(f"obs_names={list(a.obs_names) == list(b.obs_names)}")
    d.append(f"var_names={list(a.var_names) == list(b.var_names)}")
    try: d.append(f"obs={a.obs.equals(b.obs)}")
    except Exception as e: d.append(f"obs=ERR{e}")
    try: d.append(f"var={a.var.equals(b.var)}")
    except Exception as e: d.append(f"var=ERR{e}")
    ra = a.raw.X if a.raw is not None else None; rb = b.raw.X if b.raw is not None else None
    d.append(f"raw={same_mat(ra, rb)}")
    d.append(f"uns_keys={sorted(a.uns.keys()) == sorted(b.uns.keys())}")
    ok = all(s.endswith('=True') for s in d)
    return [name, ma, mb, 'CONTENT-IDENTICAL' if ok else 'DIFFERS', ';'.join(d) + f";shape={a.shape}vs{b.shape}"]

if __name__ == '__main__':
    A, B, out = sys.argv[1], sys.argv[2], sys.argv[3]
    names = sorted(p.name for p in Path(A).glob('*.h5ad'))
    with Pool(int(sys.argv[4]) if len(sys.argv) > 4 else 8) as pool:
        rows = pool.map(cmp, [(n, A, B) for n in names if (Path(B) / n).exists()])
    onlyA = [n for n in names if not (Path(B) / n).exists()]
    with open(out, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t'); w.writerow(['file', 'md5_repro', 'md5_canonical', 'result', 'detail'])
        w.writerows(rows)
        for n in onlyA: w.writerow([n, '', '', 'ONLY_IN_REPRO', ''])
    from collections import Counter
    print(Counter(r[3] for r in rows), 'only_in_repro', len(onlyA))
