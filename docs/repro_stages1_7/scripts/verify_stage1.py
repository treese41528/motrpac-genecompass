"""Stage 1 check: every file in config/corpus_v2_plan.tsv exists in the canonical QC dir and matches the plan's md5 and n_cells."""
import csv, sys, hashlib, h5py
from pathlib import Path
from multiprocessing import Pool
QC = Path('/depot/reese18/apps/motrpac-genecompass/data/training/qc_h5ad')
def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()
def one(r):
    p = QC / f"{r['sample_id']}.h5ad"
    if not p.exists(): return [r['sample_id'], r['action'], r['md5'], '', r['n_cells'], '', 'MISSING']
    m = md5(p)
    try:
        with h5py.File(p, 'r') as f:
            idx = f['obs'].attrs.get('_index', '_index'); n = f['obs'][idx].shape[0]
    except Exception as e: n = f'ERR {e}'
    ok = (m == r['md5']) and (str(n) == str(r['n_cells']))
    return [r['sample_id'], r['action'], r['md5'], m, r['n_cells'], n, 'MATCH' if ok else ('MD5_MISMATCH' if m != r['md5'] else 'NCELLS_MISMATCH')]
rows = list(csv.DictReader(open('/depot/reese18/apps/motrpac-genecompass/config/corpus_v2_plan.tsv'), delimiter='\t'))
with Pool(int(sys.argv[2])) as pool: res = pool.map(one, rows, chunksize=1)
with open(sys.argv[1], 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t'); w.writerow(['sample_id', 'action', 'md5_plan', 'md5_now', 'n_cells_plan', 'n_cells_now', 'result']); w.writerows(res)
from collections import Counter
print(Counter((r[1].startswith('keep'), r[6]) for r in res))
print('kept cells now:', sum(int(r[5]) for r in res if r[1].startswith('keep') and isinstance(r[5], int)))
