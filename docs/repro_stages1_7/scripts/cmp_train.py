"""Compare the restarted phase-1 run's per-step losses (stdout dict lines, rank 0) with v2's logged log_history."""
import ast, json, sys, csv
import numpy as np
log, state, out = sys.argv[1], sys.argv[2], sys.argv[3]
ids, losses = [], []
for line in open(log):
    if line.startswith("{'id_loss'"): ids.append(ast.literal_eval(line.strip()))
    elif line.startswith("{'loss'"): losses.append(ast.literal_eval(line.strip()))
lh = json.load(open(state))['log_history']
oid = [e for e in lh if 'id_loss' in e]; olo = [e for e in lh if 'loss' in e]
n = min(len(ids), len(oid)); m = min(len(losses), len(olo))
rows = []
for i in range(n):
    a, b = ids[i], oid[i]
    rows.append(['step_id', b['step'], b['id_loss'], a['id_loss'], a['id_loss'] - b['id_loss'], b['value_loss'], a['value_loss'], a['value_loss'] - b['value_loss']])
for i in range(m):
    a, b = losses[i], olo[i]
    rows.append(['loss50', b['step'], b['loss'], a['loss'], a['loss'] - b['loss'], b['grad_norm'], a['grad_norm'], a['learning_rate'] - b['learning_rate']])
with open(out, 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['kind', 'step', 'v2_logged', 'repro', 'diff', 'v2_logged_2', 'repro_2', 'diff_2(value_loss or lr)'])
    w.writerows(rows)
d_id = np.array([r[4] for r in rows if r[0] == 'step_id']); d_v = np.array([r[7] for r in rows if r[0] == 'step_id'])
d_l = np.array([r[4] for r in rows if r[0] == 'loss50']); d_lr = np.array([r[7] for r in rows if r[0] == 'loss50'])
exact = int((d_id == 0).sum()); first_diff = next((rows[i][1] for i in range(n) if rows[i][4] != 0 or rows[i][7] != 0), None)
print(f"per-step id/value entries compared: {n}; exactly equal id_loss: {exact}; first step with any difference: {first_diff}")
print(f"|d id_loss| max {np.abs(d_id).max():.4g} median {np.median(np.abs(d_id)):.4g}; |d value_loss| max {np.abs(d_v).max():.4g}")
print(f"50-step mean loss entries: {m}; |d loss| max {np.abs(d_l).max():.4g} median {np.median(np.abs(d_l)):.4g}; lr identical: {bool((d_lr == 0).all())}")
for blk in range(0, n, 500):
    sl = slice(blk, min(blk + 500, n))
    a = np.array([r[3] for r in rows[sl]]); b = np.array([r[2] for r in rows[sl]])
    print(f"steps {blk}-{min(blk+500,n)-1}: mean id_loss v2 {b.mean():.4f} repro {a.mean():.4f}; corr {np.corrcoef(a, b)[0,1]:.4f}; exact {int((a == b).sum())}")
