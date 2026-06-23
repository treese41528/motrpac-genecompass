#!/usr/bin/env python3
"""
compare_posctrl.py -- execute the FROZEN positive-control pre-registration against
the exhaustive per-cell-type DE outputs.

This runs ONLY the committed spec (deconvolution/reference/posctrl_prereg.tsv, frozen by
build_posctrl_prereg.py BEFORE any per-gene result was seen). It does NOT scan for new genes.
See deconvolution/POSCTRL_PREREG.md for the design, scoring rules, and the frozen miss-ladder.

Scoring (frozen):
  Tier A  (anchored_direction): RECOVERED iff testable + powered + dose-significant +
          sign(8w logFC) matches expected_dir. Else wrong_direction / not_significant /
          underpowered / not_testable.
  Tier Ai,B (identity): RECOVERED iff testable + powered + significant in the matching
          tissue (+sex); direction reported, not pass/fail.
  Tier C  (Yu programs): cell-type RESPONSIVENESS only -- per (program, cell type) the
          fraction of testable program genes that are training-DE vs the block-wide DE rate
          (one-sided binomial enrichment). Direction NOT scored.

Significance: sex=any -> FDR_IHW < alpha (global IHW~tissue dose FDR);
              sex=male/female -> per-sex 8w BH (fdr_8w_M / fdr_8w_F) < alpha.
Power floor:  cell-type mean_fraction >= --min-fraction AND gene n_nonzero >= --min-nonzero.
Confound:     block frac_week_p < alpha -> read relative/differential (flagged, not excluded).

Miss-ladder (frozen order): coverage (testable?) -> power -> confound -> biology.

Outputs (under --out): posctrl_results.tsv (per resolved gene/program row) and
posctrl_summary.md (tallies per tier/group + the pre-stated-expectation verdict).
"""
import argparse
import math
import os
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np

_PROJECT_ROOT = Path(os.environ.get('PIPELINE_ROOT') or Path(__file__).resolve().parents[1])


def safe(s: str) -> str:
    """Mirror the R safe(): gsub('[^A-Za-z0-9]+','_')."""
    return re.sub(r'[^A-Za-z0-9]+', '_', s)


def binom_upper_tail(m: int, k: int, p: float) -> float:
    """P(X >= m) for X ~ Binom(k, p). Exact (k is small here)."""
    if k <= 0:
        return float('nan')
    p = min(max(p, 1e-12), 1 - 1e-12)
    return float(sum(math.comb(k, i) * p**i * (1 - p)**(k - i) for i in range(m, k + 1)))


class DEStore:
    """Lazy loader/cache for de_summary.tsv and per-block de__*.tsv tables."""
    REQUIRED = {'gene', 'n_nonzero', 'FDR_IHW', 'lfc_8w', 'lfc_8w_M', 'lfc_8w_F',
                'fdr_8w_M', 'fdr_8w_F'}

    def __init__(self, de_dir: Path, alpha: float):
        self.de_dir = de_dir
        self.alpha = alpha
        sf = de_dir / 'de_summary.tsv'
        if not sf.exists():
            sys.exit(f"ERROR: de_summary.tsv not found under {de_dir}\n"
                     f"  -> run the exhaustive DE first (Stage 10 step 1 / run_pseudobulk_de.sh)")
        self.summary = pd.read_csv(sf, sep='\t')
        # validate exhaustive schema on a sample block (guards against running on the old linear-only outputs)
        self._cache = {}
        self._block_rate = {}

    def tissues(self):
        return sorted(self.summary['tissue'].astype(str).unique())

    def celltypes(self, tissue: str):
        s = self.summary[(self.summary.tissue == tissue) & (self.summary.status == 'ok')]
        return list(s['cell_type'].astype(str))

    def block_meta(self, tissue: str, ct: str):
        s = self.summary[(self.summary.tissue == tissue) & (self.summary.cell_type == ct)]
        if s.empty:
            return None
        r = s.iloc[0]
        return dict(mean_fraction=float(r.get('mean_fraction', np.nan)),
                    frac_week_p=float(r.get('frac_week_p', np.nan)),
                    status=str(r.get('status', '')),
                    n_genes_tested=int(r.get('n_genes_tested', 0) or 0))

    def block(self, tissue: str, ct: str):
        key = (tissue, ct)
        if key in self._cache:
            return self._cache[key]
        f = self.de_dir / tissue / f"de__{safe(ct)}.tsv"
        df = None
        if f.exists():
            df = pd.read_csv(f, sep='\t')
            missing = self.REQUIRED - set(df.columns)
            if missing:
                sys.exit(f"ERROR: {f} is missing columns {sorted(missing)} -- these DE outputs are not "
                         f"the exhaustive schema. Re-run the exhaustive run_pseudobulk_de.R (Stage 10 step 1).")
            df = df.set_index('gene')
        self._cache[key] = df
        return df

    def block_de_rate(self, tissue: str, ct: str):
        key = (tissue, ct)
        if key in self._block_rate:
            return self._block_rate[key]
        df = self.block(tissue, ct)
        rate = float('nan')
        if df is not None and 'FDR_IHW' in df:
            fin = df['FDR_IHW'].notna()
            if fin.sum() > 0:
                rate = float((df.loc[fin, 'FDR_IHW'] < self.alpha).mean())
        self._block_rate[key] = rate
        return rate


def gene_eval(store, tissue, ct, ensembl, sex, min_fraction, min_nonzero, alpha):
    """Evaluate one gene in one (tissue, cell-type) block. Returns a dict or None if block missing."""
    df = store.block(tissue, ct)
    meta = store.block_meta(tissue, ct)
    if df is None or meta is None:
        return None
    present = ensembl in df.index
    out = dict(cell_type=ct, present=present, mean_fraction=meta['mean_fraction'],
               frac_week_p=meta['frac_week_p'], confounded=(meta['frac_week_p'] < alpha)
               if pd.notna(meta['frac_week_p']) else False,
               n_nonzero=np.nan, FDR_IHW=np.nan, lfc=np.nan, sig=False, powered=False, direction=0)
    if not present:
        return out
    row = df.loc[ensembl]
    if isinstance(row, pd.DataFrame):           # duplicate gene id (shouldn't happen) -> take first
        row = row.iloc[0]
    nnz = float(row.get('n_nonzero', np.nan))
    out['n_nonzero'] = nnz
    out['FDR_IHW'] = float(row.get('FDR_IHW', np.nan))
    out['powered'] = (pd.notna(meta['mean_fraction']) and meta['mean_fraction'] >= min_fraction
                      and pd.notna(nnz) and nnz >= min_nonzero)
    if sex == 'male':
        sig_p, lfc = row.get('fdr_8w_M', np.nan), row.get('lfc_8w_M', np.nan)
    elif sex == 'female':
        sig_p, lfc = row.get('fdr_8w_F', np.nan), row.get('lfc_8w_F', np.nan)
    else:
        sig_p, lfc = row.get('FDR_IHW', np.nan), row.get('lfc_8w', np.nan)
    out['lfc'] = float(lfc) if pd.notna(lfc) else np.nan
    out['sig'] = bool(pd.notna(sig_p) and float(sig_p) < alpha)
    out['direction'] = int(np.sign(out['lfc'])) if pd.notna(out['lfc']) else 0
    return out


def resolve_targets(store, tissue, target):
    if target.strip().upper() == 'ANY':
        return store.celltypes(tissue)
    wanted = [t.strip() for t in target.split('|')]
    avail = set(store.celltypes(tissue))
    return [t for t in wanted if t in avail]


def classify_identity(evals, tier, expected_dir):
    """Aggregate per-cell-type evals for a single gene into one frozen-ladder outcome.
    evals: list of gene_eval dicts (present blocks only considered for the ladder)."""
    present = [e for e in evals if e['present']]
    if not present:
        return 'not_testable', None
    powered = [e for e in present if e['powered']]
    if not powered:
        # present but no powered block -> power rung
        best = max(present, key=lambda e: (e['mean_fraction'] if pd.notna(e['mean_fraction']) else -1))
        return 'underpowered', best
    sig = [e for e in powered if e['sig']]
    if not sig:
        best = max(powered, key=lambda e: (e['mean_fraction'] if pd.notna(e['mean_fraction']) else -1))
        return 'not_significant', best          # covered + powered + flat -> biology rung
    if tier == 'A' and expected_dir in ('up', 'down'):
        want = 1 if expected_dir == 'up' else -1
        dir_ok = [e for e in sig if e['direction'] == want]
        if dir_ok:
            e = max(dir_ok, key=lambda e: (e['mean_fraction'] if pd.notna(e['mean_fraction']) else -1))
            return ('recovered_confounded' if e['confounded'] else 'recovered'), e
        e = sig[0]
        return 'wrong_direction', e
    # identity tiers (Ai, B): significant is enough
    e = max(sig, key=lambda e: (e['mean_fraction'] if pd.notna(e['mean_fraction']) else -1))
    return ('recovered_confounded' if e['confounded'] else 'recovered'), e


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--de-dir', default=str(_PROJECT_ROOT / 'data/deconvolution/genecompass_input/pseudobulk_de'))
    ap.add_argument('--prereg', default=str(_PROJECT_ROOT / 'deconvolution/reference/posctrl_prereg.tsv'))
    ap.add_argument('--out', default=None, help='output dir (default: --de-dir)')
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--min-fraction', type=float, default=0.01)
    ap.add_argument('--min-nonzero', type=int, default=25)
    args = ap.parse_args()

    de_dir = Path(args.de_dir)
    out_dir = Path(args.out) if args.out else de_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prereg = pd.read_csv(args.prereg, sep='\t')
    store = DEStore(de_dir, args.alpha)

    rows = []            # per-gene (tiers A/Ai/B) result rows
    cprog = []           # Tier C program-enrichment rows

    # --- tiers A / Ai / B : per-gene identity/direction ---
    idmask = prereg.tier.isin(['A', 'Ai', 'B'])
    for r in prereg[idmask].itertuples():
        base = dict(tier=r.tier, group=r.group, symbol=r.symbol, ensembl=r.ensembl,
                    tissue=r.tissue, sex=r.sex, expected_dir=r.expected_dir)
        if str(r.testable_prior).upper() == 'FALSE' or str(r.ensembl) in ('NA', 'nan', ''):
            rows.append({**base, 'cell_type': '', 'outcome': 'not_testable_prior',
                         'present': False, 'n_nonzero': np.nan, 'mean_fraction': np.nan,
                         'FDR_IHW': np.nan, 'lfc_8w': np.nan, 'frac_week_p': np.nan,
                         'powered': False, 'significant': False, 'direction_match': ''})
            continue
        targets = resolve_targets(store, r.tissue, r.cell_type_target)
        evals = []
        for ct in targets:
            e = gene_eval(store, r.tissue, ct, r.ensembl, r.sex, args.min_fraction, args.min_nonzero, args.alpha)
            if e is not None:
                evals.append(e)
        outcome, e = classify_identity(evals, r.tier, r.expected_dir)
        if e is None:
            rows.append({**base, 'cell_type': '|'.join(targets) or '(no target blocks)',
                         'outcome': outcome, 'present': False, 'n_nonzero': np.nan,
                         'mean_fraction': np.nan, 'FDR_IHW': np.nan, 'lfc_8w': np.nan,
                         'frac_week_p': np.nan, 'powered': False, 'significant': False, 'direction_match': ''})
        else:
            dmatch = ''
            if r.tier == 'A' and r.expected_dir in ('up', 'down'):
                want = 1 if r.expected_dir == 'up' else -1
                dmatch = str(e['direction'] == want)
            rows.append({**base, 'cell_type': e['cell_type'], 'outcome': outcome,
                         'present': e['present'], 'n_nonzero': e['n_nonzero'],
                         'mean_fraction': e['mean_fraction'], 'FDR_IHW': e['FDR_IHW'],
                         'lfc_8w': e['lfc'], 'frac_week_p': e['frac_week_p'],
                         'powered': e['powered'], 'significant': e['sig'], 'direction_match': dmatch})

    # --- tier C : program responsiveness (enrichment), direction NOT scored ---
    tc = prereg[prereg.tier == 'C']
    for (group, tissue), g in tc.groupby(['group', 'tissue']):
        target = g['cell_type_target'].iloc[0]
        ens = [e for e in g['ensembl'].tolist() if str(e) not in ('NA', 'nan', '')]
        for ct in resolve_targets(store, tissue, target):
            df = store.block(tissue, ct)
            meta = store.block_meta(tissue, ct)
            if df is None or meta is None:
                continue
            testable = [e for e in ens if e in df.index]
            k = len(testable)
            hits = [e for e in testable if pd.notna(df.loc[e, 'FDR_IHW']) and df.loc[e, 'FDR_IHW'] < args.alpha]
            m = len(hits)
            rate = store.block_de_rate(tissue, ct)
            cprog.append(dict(tier='C', group=group, tissue=tissue, cell_type=ct,
                              n_program_genes=len(ens), n_testable=k, n_DE=m,
                              program_DE_rate=(m / k if k else float('nan')),
                              block_DE_rate=rate,
                              enrichment=((m / k) / rate) if (k and rate and rate > 0) else float('nan'),
                              binom_p=binom_upper_tail(m, k, rate) if (k and pd.notna(rate)) else float('nan'),
                              mean_fraction=meta['mean_fraction'], frac_week_p=meta['frac_week_p']))

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / 'posctrl_results.tsv', sep='\t', index=False)
    prog = pd.DataFrame(cprog)
    prog.to_csv(out_dir / 'posctrl_responsiveness.tsv', sep='\t', index=False)

    # --- human-readable summary ---
    def tally(df, tiers):
        sub = df[df.tier.isin(tiers)]
        return sub['outcome'].value_counts().to_dict(), len(sub)

    lines = []
    lines.append("# Positive-control comparison (executes the frozen pre-registration)\n")
    lines.append(f"DE dir: `{de_dir}`  |  pre-reg: `{Path(args.prereg).name}`  |  "
                 f"alpha={args.alpha}, power floor: mean_fraction>={args.min_fraction} & n_nonzero>={args.min_nonzero}\n")
    lines.append("Outcomes follow the frozen miss-ladder: not_testable(_prior) -> underpowered -> "
                 "not_significant(biology) -> wrong_direction / recovered(_confounded).\n")
    for tier, label in [('A', 'Tier A -- direction-anchored (MoTrPAC main)'),
                        ('Ai', 'Tier Ai -- MoTrPAC identity-only'),
                        ('B', 'Tier B -- Vetr named (identity+tissue+sex)')]:
        counts, n = tally(res, [tier])
        rec = counts.get('recovered', 0) + counts.get('recovered_confounded', 0)
        lines.append(f"\n## {label}  ({rec}/{n} recovered)\n")
        for k in ['recovered', 'recovered_confounded', 'wrong_direction', 'not_significant',
                  'underpowered', 'not_testable', 'not_testable_prior']:
            if counts.get(k):
                lines.append(f"  - {k}: {counts[k]}")
        sub = res[res.tier == tier].sort_values('outcome')
        lines.append("\n| symbol | tissue | cell_type | sex | exp_dir | FDR_IHW | lfc_8w | outcome | dir_match |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for x in sub.itertuples():
            f = (f"{x.FDR_IHW:.2g}" if pd.notna(x.FDR_IHW) else "NA")
            l = (f"{x.lfc_8w:+.2f}" if pd.notna(x.lfc_8w) else "NA")
            lines.append(f"| {x.symbol} | {x.tissue} | {x.cell_type} | {x.sex} | {x.expected_dir} "
                         f"| {f} | {l} | {x.outcome} | {x.direction_match} |")

    lines.append("\n## Tier C -- Yu immune programs (responsiveness; direction NOT scored)\n")
    if not prog.empty:
        lines.append("| program | tissue | cell_type | n_DE/n_testable | program_rate | block_rate | enrichment | binom_p |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for x in prog.sort_values(['group', 'enrichment'], ascending=[True, False]).itertuples():
            enr = (f"{x.enrichment:.2f}" if pd.notna(x.enrichment) else "NA")
            bp = (f"{x.binom_p:.2g}" if pd.notna(x.binom_p) else "NA")
            lines.append(f"| {x.group} | {x.tissue} | {x.cell_type} | {x.n_DE}/{x.n_testable} "
                         f"| {x.program_DE_rate:.3f} | {x.block_DE_rate:.3f} | {enr} | {bp} |")
    else:
        lines.append("(no Tier-C target blocks found)")

    lines.append("\n## Verdict vs pre-stated expectations\n")
    aA = res[res.tier == 'A']
    musc = aA[(aA.group.str.contains('mito')) &
              (aA.outcome.isin(['recovered', 'recovered_confounded']))]
    lines.append(f"  - PIPELINE SANITY (Tier-A muscle/heart mito+heat-shock UP): "
                 f"{len(musc)}/{len(aA[aA.group.str.contains('mito')])} recovered direction-concordant. "
                 f"(Strong non-recovery here = pipeline red flag, not biology.)")
    lines.append(f"  - Full per-gene detail: posctrl_results.tsv ; program detail: posctrl_responsiveness.tsv")

    (out_dir / 'posctrl_summary.md').write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir/'posctrl_results.tsv'} ({len(res)} rows), "
          f"{out_dir/'posctrl_responsiveness.tsv'} ({len(prog)} rows), {out_dir/'posctrl_summary.md'}")
    # console digest
    for tier in ['A', 'Ai', 'B']:
        counts, n = tally(res, [tier])
        rec = counts.get('recovered', 0) + counts.get('recovered_confounded', 0)
        print(f"  Tier {tier}: {rec}/{n} recovered | " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == '__main__':
    main()
