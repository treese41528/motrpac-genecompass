"""Compare a rebuilt rat init checkpoint with the canonical v2 init, tensor by tensor."""
import sys, csv, torch
a = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
b = torch.load(sys.argv[2], map_location='cpu', weights_only=False)
tier = {}
for r in csv.DictReader(open(sys.argv[4]), delimiter='\t'):
    tier.setdefault(int(r['token_id']), set()).add(r['tier'][:2])
rows = []
for k in sorted(set(a) | set(b)):
    if k not in a or k not in b: rows.append([k, 'MISSING', '', '']); continue
    x, y = a[k], b[k]
    if not torch.is_tensor(x): rows.append([k, 'EQUAL' if x == y else 'DIFF', '', '']); continue
    if x.shape != y.shape: rows.append([k, 'SHAPE', f'{tuple(x.shape)} vs {tuple(y.shape)}', '']); continue
    if torch.equal(x, y): rows.append([k, 'EQUAL', tuple(x.shape), '']); continue
    d = (x.float() - y.float()).abs()
    det = f'max|d|={d.max().item():.3g}'
    if d.dim() == 2:
        bad = torch.nonzero(d.amax(1) > 0).flatten().tolist()
        groups = {}
        for i in bad:
            g = 'orig(<50558)' if i < 50558 else '+'.join(sorted(tier.get(i, {'?'})))
            groups[g] = groups.get(g, 0) + 1
        det += f'; rows differing={len(bad)} by group={groups}'
        if bad and min(bad) >= 50558:
            for g in sorted(groups):
                ids = [i for i in bad if ('+'.join(sorted(tier.get(i, {'?'}))) == g)]
                det += f'; {g} max|d|={d[ids].max().item():.3g} mean row-std repro={x[ids].float().std().item():.3g} canon={y[ids].float().std().item():.3g}'
    rows.append([k, 'DIFF', tuple(x.shape), det])
with open(sys.argv[3], 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t'); w.writerow(['tensor', 'result', 'shape', 'detail']); w.writerows(rows)
for r in rows:
    if r[1] != 'EQUAL': print(r)
print('equal tensors:', sum(r[1] == 'EQUAL' for r in rows), 'of', len(rows))
