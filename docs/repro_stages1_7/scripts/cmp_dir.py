"""Parallel compare of same-named files in two directories (non-recursive).
md5 first; on mismatch: .arrow -> table content (pyarrow), .json -> volatile keys stripped, .pickle -> unpickled equality,
.tsv/.txt -> sorted-line equality, .npz -> arrays equal."""
import sys, hashlib, json, csv, pickle
from pathlib import Path
from multiprocessing import Pool
VOL = {'timestamp', 'generated_at', 'generated', 'elapsed_seconds', 'elapsed_s', 'git_hash', 'git_commit', 'created',
       'host', 'profile', 'git_modified_tracked_files', 'mtime', 'cwd', '_fingerprint'}
A = B = None
def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()
def strip(o):
    if isinstance(o, dict): return {k: strip(v) for k, v in o.items() if k not in VOL}
    if isinstance(o, list): return [strip(v) for v in o]
    if isinstance(o, str): return o.replace(A, '<DIR>').replace(B, '<DIR>')
    return o
def diffpaths(a, b, path=''):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b: out.append(f"{path}/{k}:missing")
            else: out += diffpaths(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)): out += diffpaths(x, y, f"{path}[{i}]")
    elif a != b: out.append(f"{path}: {str(a)[:100]} != {str(b)[:100]}")
    return out
def read_arrow(p):
    import pyarrow as pa
    with pa.memory_map(str(p)) as src:
        try: return pa.ipc.open_stream(src).read_all()
        except Exception:
            src.seek(0); return pa.ipc.open_file(src).read_all()
def content(n, pa_, pb_):
    try:
        if n.endswith('.arrow'):
            ta, tb = read_arrow(pa_), read_arrow(pb_)
            if ta.schema.remove_metadata().equals(tb.schema.remove_metadata()) and ta.num_rows == tb.num_rows:
                eq = ta.select(ta.column_names).cast(ta.schema).equals(tb.select(ta.column_names).cast(ta.schema)) if set(ta.column_names) == set(tb.column_names) else False
                if eq: return 'CONTENT-IDENTICAL', f'rows={ta.num_rows}; schema metadata equal={ta.schema.equals(tb.schema, check_metadata=True)}'
                cols = [c for c in ta.column_names if c in tb.column_names and not ta.column(c).equals(tb.column(c))]
                return 'DIFFERS', f'rows={ta.num_rows}; differing columns={cols}'
            return 'DIFFERS', f'schema/rows: {ta.num_rows} vs {tb.num_rows}; {ta.schema.names} vs {tb.schema.names}'
        if n.endswith('.json'):
            d = diffpaths(strip(json.load(open(pa_))), strip(json.load(open(pb_))))
            return ('CONTENT-IDENTICAL' if not d else 'DIFFERS'), f"{len(d)} diffs: " + ' | '.join(d[:6])
        if n.endswith('.pickle') or n.endswith('.pkl'):
            import numpy as np
            a, b = pickle.load(open(pa_, 'rb')), pickle.load(open(pb_, 'rb'))
            try: eq = a == b; eq = bool(eq) if not hasattr(eq, 'all') else bool(eq.all())
            except Exception:
                eq = type(a) == type(b) and isinstance(a, dict) and a.keys() == b.keys() and all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a)
            return ('CONTENT-IDENTICAL' if eq else 'DIFFERS'), f'type={type(a).__name__} len={len(a) if hasattr(a, "__len__") else ""}'
        if n.endswith('.npz'):
            import numpy as np
            a, b = np.load(pa_, allow_pickle=True), np.load(pb_, allow_pickle=True)
            bad = [k for k in sorted(set(a.files) | set(b.files)) if k not in a.files or k not in b.files or not np.array_equal(a[k], b[k])]
            return ('CONTENT-IDENTICAL' if not bad else 'DIFFERS'), f'differing arrays={bad}'
        if n.endswith('.gz'):
            import gzip
            la, lb = gzip.open(pa_, 'rt').read(), gzip.open(pb_, 'rt').read()
            return ('CONTENT-IDENTICAL' if la == lb else 'DIFFERS'), 'decompressed text compared'
        if n.endswith(('.tsv', '.txt', '.csv')):
            la, lb = open(pa_).read().splitlines(), open(pb_).read().splitlines()
            if la == lb: return 'CONTENT-IDENTICAL', ''
            return 'DIFFERS', f'lines {len(la)} vs {len(lb)}; same set={sorted(la) == sorted(lb)}; first diff line={next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), None)}'
    except Exception as e:
        return 'DIFFERS', f'compare error {e!r}'
    return 'DIFFERS', ''
def one(n):
    pa_, pb_ = Path(A) / n, Path(B) / n
    if not pb_.exists(): return [n, md5(pa_), '', 'ONLY_IN_REPRO', '']
    ma, mb = md5(pa_), md5(pb_)
    if ma == mb: return [n, ma, mb, 'IDENTICAL', '']
    r, d = content(n, pa_, pb_)
    return [n, ma, mb, r, d]
if __name__ == '__main__':
    A, B, out = sys.argv[1], sys.argv[2], sys.argv[3]
    nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    names = sorted(p.name for p in Path(A).iterdir() if p.is_file())
    with Pool(nproc, initializer=lambda a, b: globals().update(A=a, B=b), initargs=(A, B)) as pool:
        rows = pool.map(one, names, chunksize=1)
    for n in sorted(p.name for p in Path(B).iterdir() if p.is_file()):
        if not (Path(A) / n).exists(): rows.append([n, '', '', 'ONLY_IN_CANONICAL', ''])
    with open(out, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t'); w.writerow(['file', 'md5_repro', 'md5_canonical', 'result', 'detail']); w.writerows(rows)
    from collections import Counter
    print(out, Counter(r[3] for r in rows))
    for r in rows:
        if r[3] not in ('IDENTICAL',): print('  ', '\t'.join(r[:1] + r[3:])[:400])
