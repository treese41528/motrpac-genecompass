"""md5-compare same-named files in two dirs; for JSON, also compare with volatile keys removed."""
import sys, hashlib, json, csv
from pathlib import Path
VOL = {'timestamp', 'generated_at', 'generated', 'elapsed_seconds', 'elapsed_s', 'git_hash', 'git_commit', 'created',
       'host', 'profile', 'git_modified_tracked_files', 'mtime', 'cwd'}
def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()
def strip(o, A, B):
    if isinstance(o, dict): return {k: strip(v, A, B) for k, v in o.items() if k not in VOL}
    if isinstance(o, list): return [strip(v, A, B) for v in o]
    if isinstance(o, str): return o.replace(A, '<DIR>').replace(B, '<DIR>')
    return o
def diffpaths(a, b, path=''):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b: out.append(f"{path}/{k}:missing")
            else: out += diffpaths(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)): out += diffpaths(x, y, f"{path}[{i}]")
    elif a != b: out.append(f"{path}: {str(a)[:80]} != {str(b)[:80]}")
    return out
A, B, out = sys.argv[1], sys.argv[2], sys.argv[3]
names = sorted(p.name for p in Path(A).iterdir() if p.is_file())
rows = []
for n in names:
    pa, pb = Path(A) / n, Path(B) / n
    if not pb.exists(): rows.append([n, md5(pa), '', 'ONLY_IN_REPRO', '']); continue
    ma, mb = md5(pa), md5(pb)
    if ma == mb: rows.append([n, ma, mb, 'IDENTICAL', '']); continue
    det, res = '', 'DIFFERS'
    if n.endswith('.json'):
        ja, jb = strip(json.load(open(pa)), A, B), strip(json.load(open(pb)), A, B)
        d = diffpaths(ja, jb)
        res = 'CONTENT-IDENTICAL' if not d else 'DIFFERS'
        det = f"{len(d)} diffs: " + ' | '.join(d[:8])
    rows.append([n, ma, mb, res, det])
for n in sorted(p.name for p in Path(B).iterdir() if p.is_file()):
    if not (Path(A) / n).exists(): rows.append([n, '', md5(Path(B) / n), 'ONLY_IN_CANONICAL', ''])
with open(out, 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t'); w.writerow(['file', 'md5_repro', 'md5_canonical', 'result', 'detail']); w.writerows(rows)
for r in rows: print('\t'.join(r)[:300])
