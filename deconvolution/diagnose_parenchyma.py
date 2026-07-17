#!/usr/bin/env python3
"""
diagnose_parenchyma.py -- why do the dominant-parenchyma cell types FAIL the Tier-A bulk
positive controls (Sod2/Mef2c/Slc2a4/Hspa1b/Hsp90aa1) in the per-cell-type DE while producing
hundreds of OTHER dose-significant genes?

Tests three hypotheses by comparing the SEX-ADJUSTED ordinal-week dose-slope of each gene in
(1) the BULK matrix vs (2) the parenchyma cell type's deconvolved Z:
  H3 (compression/prior-regression): genome-wide Z-slope CORRELATES with bulk-slope but is
      SHRUNK in magnitude (shrinkage<1) -> small-bulk-slope controls fall below detection.
  decoupling (worse): Z-slope does NOT correlate with bulk-slope -> Z carries a different signal.
  H1 (over-split): for SKMVL only, pooling the two muscle-parenchyma labels recovers the trend.

Read-only, vectorized, seconds-scale. No model, no GPU.
"""
import os, re, sys
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy import stats

ROOT = os.environ.get('PIPELINE_ROOT', '.')
BULK = os.path.join(ROOT, 'data/deconvolution/motrpac_bulk')
RES  = os.path.join(ROOT, 'data/deconvolution/results/motrpac')
PHENO = os.path.join(ROOT, 'deconvolution/reference/motrpac_sample_pheno.tsv')
WEEK = {'control': 0, '1w': 1, '2w': 2, '4w': 4, '8w': 8}
CTRL = {'Sod2': 'ENSRNOG00000086727', 'Mef2c': 'ENSRNOG00000033134', 'Slc2a4': 'ENSRNOG00000017226',
        'Hspa1b': 'ENSRNOG00000045654', 'Hsp90aa1': 'ENSRNOG00000059714',
        'Opa1': 'ENSRNOG00000001717'}   # Opa1 = a control that DID recover (reference point)
PAR = {'SKMGN': ['Skeletal myocytes'],
       'SKMVL': ['Skeletal myocytes'],
       'HEART': ['Cardiomyocytes'],
       'LIVER': ['Hepatocytes']}

from celltype_names import safe  # noqa: E402  (shared writer/reader filename contract)

ph = pd.read_csv(PHENO, sep='\t', dtype=str).drop_duplicates('viallabel').set_index('viallabel')

def week_slopes(logmat, week, sex):
    """Sex-adjusted week slope per gene (rows=genes, cols=samples). Vectorized OLS week coef."""
    X = np.column_stack([np.ones_like(week, float), week.astype(float), sex.astype(float)])
    B = np.linalg.pinv(X) @ logmat.T          # (3 x genes)
    return B[1]                                # week coefficient per gene

def log2cpm(counts):                            # counts: genes x samples
    cs = counts.sum(0); cs[cs == 0] = 1
    return np.log2(counts / cs * 1e6 + 1)

for TIS in ['SKMGN', 'SKMVL', 'HEART', 'LIVER']:
    print('=' * 78); print(f'[{TIS}]')
    via = [x.strip() for x in open(os.path.join(BULK, TIS, 'bulk_samples.tsv')) if x.strip()]
    bgenes = [x.strip() for x in open(os.path.join(BULK, TIS, 'bulk_genes.tsv')) if x.strip()]
    M = np.asarray(mmread(os.path.join(BULK, TIS, 'bulk.mtx')).todense(), float)
    if M.shape[0] != len(bgenes):              # orient to genes x samples
        M = M.T
    week = np.array([WEEK.get(ph['group'].get(v, ''), np.nan) for v in via], float)
    sexv = np.array([1.0 if ph['sex'].get(v, '') == 'male' else 0.0 for v in via])
    ok = ~np.isnan(week)
    M, week_, sex_ = M[:, ok], week[ok], sexv[ok]
    blog = log2cpm(M)
    bslope = week_slopes(blog, week_, sex_)
    bidx = {g: i for i, g in enumerate(bgenes)}

    # parenchyma Z (sum across labels for the over-split pool; also per-label)
    pz_each = {}
    for ct in PAR[TIS]:
        f = os.path.join(RES, TIS, 'pred_z', f'predz__{safe(ct)}.csv')
        if not os.path.exists(f):
            print(f'  (missing {ct})'); continue
        Z = pd.read_csv(f, index_col=0)        # samples x genes
        pz_each[ct] = Z
    if not pz_each:
        continue
    # align on shared genes across the first label's Z and bulk
    any_ct = next(iter(pz_each))
    zgenes = list(pz_each[any_ct].columns)
    shared = [g for g in zgenes if g in bidx]
    bsl_sh = np.array([bslope[bidx[g]] for g in shared])

    def z_slopes_for(Zmat):                     # Zmat samples x genes (shared order)
        zlog = np.log2(Zmat.T.values[:, ok] + 1)   # genes x samples (ok-filtered)
        return week_slopes(zlog, week_, sex_)

    # pooled parenchyma (sum count-mass across labels)
    Zpool = sum(pz_each[ct][shared] for ct in pz_each)
    zsl_pool = z_slopes_for(Zpool)
    r, p = stats.pearsonr(bsl_sh, zsl_pool)
    shrink = np.polyfit(bsl_sh, zsl_pool, 1)[0]      # slope of Zslope ~ bulkslope
    print(f'  genome-wide (n={len(shared)} genes): corr(bulk-slope, pooled-parenchyma-Z-slope) '
          f'r={r:.3f} (p={p:.1e}); shrinkage(Zslope/bulkslope)={shrink:.3f}')
    # per single label too (over-split check)
    if len(pz_each) > 1:
        for ct in pz_each:
            zs = z_slopes_for(pz_each[ct][shared])
            rr, _ = stats.pearsonr(bsl_sh, zs)
            sh = np.polyfit(bsl_sh, zs, 1)[0]
            print(f'    label "{ct}": r={rr:.3f}, shrinkage={sh:.3f}')

    # control genes specifically
    print(f'  {"gene":9s} {"bulk_slope":>10s} {"bulk_rankpct":>12s} {"Zpool_slope":>11s} '
          f'{"sign_match":>10s} {"Z/bulk_mag":>10s}')
    rank = stats.rankdata(blog.mean(1)) / len(bslope)
    zmap = {g: i for i, g in enumerate(shared)}
    for sym, ens in CTRL.items():
        if ens not in bidx:
            print(f'  {sym:9s} (not in bulk)'); continue
        bs = bslope[bidx[ens]]
        if ens in zmap:
            zs = zsl_pool[zmap[ens]]
            mag = (zs / bs) if abs(bs) > 1e-9 else float('nan')
            sm = 'YES' if (bs * zs) > 0 else 'no'
            print(f'  {sym:9s} {bs:>10.3f} {rank[bidx[ens]]:>12.2f} {zs:>11.3f} {sm:>10s} {mag:>10.2f}')
        else:
            print(f'  {sym:9s} {bs:>10.3f} {rank[bidx[ens]]:>12.2f}  (gene not in parenchyma Z)')
print('=' * 78)
print('READ: high genome-wide r with shrinkage<1 => parenchyma Z is a COMPRESSED copy of the bulk')
print('dose signal (H3). Low r => Z decoupled. For SKMVL, compare pooled vs single-label r/shrinkage (H1).')
