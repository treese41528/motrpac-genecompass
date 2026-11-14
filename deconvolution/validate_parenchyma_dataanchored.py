#!/usr/bin/env python3
"""
A data-anchored check of the dominant cell type's DE (A) + dominant-cell-type flags (C).

A: of the genes that move with training dose in the matched bulk (8wk-vs-control, BH<0.05 &
   |log2FC|>=THR), how many does the dominant (parenchyma) cell type's DE call in the same direction
   (FDR_IHW<0.05 & sign-concordant)? This check was written after the registered positive controls
   (POSCTRL_PREREG.md) missed in muscle and heart, and it is not registered. The dominant type is most of
   its tissue, so agreement with the bulk is expected by construction; the table is a description, not a
   verdict on the pipeline, and it does not replace the registered controls (audit finding 7.3).

C: emit the dominant (most-abundant) cell type per tissue so the write-up can flag/down-weight
   absolute per-gene DE there (H3-robust reporting).

Outputs printed + written to the DE folder: $DE_DIR, else <genecompass_input_dir>/pseudobulk_de from the
config (dominant_celltype_flags.tsv, parenchyma_dataanchored_validation.tsv).
"""
import os, re, sys
import numpy as np, pandas as pd
from scipy.io import mmread
from scipy import stats

ROOT = os.environ.get('PIPELINE_ROOT', '.')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from gene_utils import load_config, resolve_path  # noqa: E402
_cfg = load_config()
BULK = str(resolve_path(_cfg, _cfg['deconvolution']['motrpac_bulk_out']))
RES  = str(resolve_path(_cfg, _cfg['deconvolution']['results_dir']) / 'motrpac')   # Stage 8 output, for the sample-key check
DE   = os.environ.get('DE_DIR') or str(resolve_path(_cfg, _cfg['deconvolution']['genecompass_input_dir']) / 'pseudobulk_de')
print(f'DE folder: {DE}')
PH   = str(resolve_path(_cfg, _cfg['deconvolution']['sample_pheno']))
THR_LFC, ALPHA, MINNZ = 0.25, 0.05, 25
# Parenchyma label per tissue; muscle is 'Skeletal muscle cells' in the jointly re-annotated reference
# (references_v4) and 'Skeletal myocytes' before it. A tissue matching none of its names is an error.
PAR = {'SKMGN': ['Skeletal myocytes', 'Skeletal muscle cells'],
       'SKMVL': ['Skeletal myocytes', 'Skeletal muscle cells'],
       'HEART': ['Cardiomyocytes'], 'LIVER': ['Hepatocytes']}
from celltype_names import safe  # noqa: E402  (shared writer/reader filename contract)
from sample_join import load_viallabels  # noqa: E402  (checked sample key, STEP3_PREREG section 2.5)
ph = pd.read_csv(PH, sep='\t', dtype=str).drop_duplicates('viallabel').set_index('viallabel')
summ = pd.read_csv(os.path.join(DE, 'de_summary.tsv'), sep='\t')

# ---- C: dominant cell type per tissue (max mean_fraction, status ok) ----
dom = (summ[summ.status == 'ok'].sort_values('mean_fraction', ascending=False)
       .groupby('tissue').head(1)[['tissue', 'cell_type', 'mean_fraction', 'n_sig_dose_IHW']])
dom = dom.rename(columns={'cell_type': 'dominant_cell_type'})
dom.to_csv(os.path.join(DE, 'dominant_celltype_flags.tsv'), sep='\t', index=False)
print('=== C: dominant (most-abundant) cell type per tissue (down-weight absolute DE here) ===')
print(dom.to_string(index=False))

# ---- A: data-anchored bulk-mover recovery in the parenchyma DE ----
print('\n=== A: do parenchyma DE blocks recover genes that GENUINELY move in the matched bulk? ===')
print('(bulk movers = 8wk-vs-control BH<0.05 & |log2FC|>=%.2f)' % THR_LFC)
rows = []
for TIS, cts in PAR.items():
    via = load_viallabels(BULK, TIS, RES)
    g = [x.strip() for x in open(f'{BULK}/{TIS}/bulk_genes.tsv') if x.strip()]
    M = np.asarray(mmread(f'{BULK}/{TIS}/bulk.mtx').todense(), float)
    if M.shape[0] != len(g): M = M.T
    cs = M.sum(0); cs[cs == 0] = 1; logc = np.log2(M / cs * 1e6 + 1)
    grp = np.array([ph['group'].get(v, '') for v in via])
    ctl, w8 = grp == 'control', grp == '8w'
    lfc = logc[:, w8].mean(1) - logc[:, ctl].mean(1)
    pv = np.array([stats.ttest_ind(logc[i, w8], logc[i, ctl], equal_var=False).pvalue
                   if (w8.sum() > 1 and ctl.sum() > 1) else 1.0 for i in range(len(g))])
    bh = pd.Series(pv).fillna(1).pipe(lambda s: s * len(s) / s.rank()).clip(upper=1).values
    mover = (bh < ALPHA) & (np.abs(lfc) >= THR_LFC)
    movers = {g[i]: lfc[i] for i in np.where(mover)[0]}
    for ct in cts:
        f = os.path.join(DE, TIS, f'de__{safe(ct)}.tsv')
        if not os.path.exists(f): continue
        d = pd.read_csv(f, sep='\t').set_index('gene')
        present = [e for e in movers if e in d.index and d.loc[e, 'n_nonzero'] >= MINNZ]
        rec = [e for e in present
               if pd.notna(d.loc[e, 'FDR_IHW']) and d.loc[e, 'FDR_IHW'] < ALPHA
               and np.sign(d.loc[e, 'lfc_8w']) == np.sign(movers[e])]
        rate = (len(rec) / len(present)) if present else float('nan')
        rows.append(dict(tissue=TIS, cell_type=ct, n_bulk_movers=len(movers),
                         n_testable=len(present), n_recovered=len(rec),
                         recovery_rate=round(rate, 3)))
res = pd.DataFrame(rows)
missing = sorted(set(PAR) - set(res.tissue if len(res) else []))
if missing:
    sys.exit(f'ERROR: no parenchyma DE block found for {missing} (renamed cell type? update PAR)')
res.to_csv(os.path.join(DE, 'parenchyma_dataanchored_validation.tsv'), sep='\t', index=False)
print(res.to_string(index=False))
# What the table is, and is not (audit finding 7.3). It is a later, unregistered check: of the genes that move
# with training in the matched bulk, how many does the dominant cell type's DE call in the same direction. The
# dominant type is most of its tissue, so its estimated expression is close to the bulk and a high rate is
# expected by construction. It does not replace the registered positive controls (POSCTRL_PREREG.md), whose
# pipeline-sanity subset is reported on its own terms. No verdict is printed: the numbers are.
tested = res[res.n_testable > 0]
print(f'\nrecovery of bulk dose movers by the dominant cell type: {int(tested.n_recovered.sum())} of '
      f'{int(tested.n_testable.sum())} testable genes over {len(tested)} tissues; per-tissue rate '
      f'{tested.recovery_rate.min():.3f} to {tested.recovery_rate.max():.3f} (median {tested.recovery_rate.median():.3f})'
      if len(tested) else '\nno tissue had a testable bulk mover')
print('This is an unregistered, later check on blocks that are most of their tissue; it carries no verdict.')
