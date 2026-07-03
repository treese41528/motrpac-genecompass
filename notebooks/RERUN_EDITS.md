# Notebook rerun edit map — pipeline8–12 after the liver+lung reference fix (2026-07-01/02)

**Context.** Two production references were wrong and have been fixed + re-run through Stage 8→12 and
adopted into production (old data backed up in `data/deconvolution/_liverlung_adoption_backup_20260701`):
- **liver** `references/liver_GSE220075` — the 2 Visium spatial samples removed (still study GSE220075).
  *Impact: negligible* (liver ~91% hepatocyte; Z 0.998-correlated to old) — liver numbers barely move.
- **lung** — engineered/in-vitro `GSE178405` **replaced** by native pooled `references/lung_native_pooled`
  (GSE273062 VeNx + GSE252844 C3 + GSE242310 NOX). *Impact: substantial* — different cell-type roster,
  fractions, and lung now shows exercise signal.

**How to use this.** The notebooks read outputs by tissue NAME, so **most cells refresh on plain
re-execution**; only the hand-copied display constants and a few stale prose conclusions need editing.
Do them in order: **(1) apply the edits below, (2) re-execute each notebook top-to-bottom.** Values I
could read from the rerun are given as OLD → NEW; a few must be pulled from the new logs (noted inline).

## Global new facts (reuse across notebooks)
| quantity | OLD | NEW | source |
|---|---|---|---|
| liver reference cells / types | 31,820 / 6 | **27,041 / 6** | `references/liver_GSE220075/summary.txt` |
| lung reference study / cells / types | GSE178405 / 54,992 / 27 | **lung_native_pooled (GSE273062+GSE252844+GSE242310) / 50,643 / 34** | `references/lung_native_pooled/summary.txt` |
| liver pseudo-cells / genes / BayesPrism min / job | 300 / 13,134 / 12.37 / 11002467 | **300 / 12,759 / 24.50 / 11170761** | `genecompass_input/liver/summary.txt`, `logs/liver_novis_casc_11170761.out` |
| lung pseudo-cells / genes / BayesPrism min / job | 1,350 / 14,929 / 17.69 / 11002632 | **1,700 / 15,348 / 14.00 / 11171098** | `genecompass_input/lung/summary.txt`, `logs/lung_native_casc_11171098.out` |
| testable blocks | 178 | **185** | `corroboration_merged.tsv` |
| exercise hotspots | 18 {blood7,heart1,kidney1,skmgn4,skmvl5} | **21 {blood7,heart1,kidney1,lung3,skmgn5,skmvl4}** | `corroboration_merged.tsv` |
| transfer preservation | PRESERVED 16 / WEAKENED 0 / LOST 2 (of 18) | **PRESERVED 19 / WEAKENED 1 / LOST 1 (of 21)** | `genecompass_input_human/transfer_comparison.tsv` |
| rerun job IDs | — | liver 11170761 · lung 11171098 · redetect 11173958 · transfer 11174632 | — |

---

## pipeline8_analysis.ipynb (Stage 8 — deconvolution)
- **CELL 0** (markdown table): LIVER row `GSE220075 | 31,820 | 6` → `GSE220075 | 27,041 | 6`; LUNG row
  `GSE178405 | 54,992 | 27` → `lung_native_pooled (GSE273062+GSE252844+GSE242310) | 50,643 | 34`. Update
  the `SLURM jobs:` header + `Executed:` date.
- **CELL 3** `timing_data` (liver=idx5, lung=idx6): liver ref_gse GSE220075 (unchanged), sc_cells
  31820→27041, prism_min 12.37→24.50, slurm_job 11002467→11170761, n_genes 13134→12759, n_pseudocells 300
  (unchanged), n_types 6 (unchanged). lung ref_gse GSE178405→`lung_native_pooled`, sc_cells 54992→50643,
  n_types 27→34, prism_min 17.69→14.00, slurm_job 11002632→11171098, n_pseudocells 1350→1700, n_genes
  14929→15348.
- **CELL 4** arrays: `sc_cells[5]` 31820→27041, `[6]` 54992→50643; `prism_min[5]` 12.37→24.50, `[6]`
  17.69→14.00.
- **CELL 7 & CELL 11** `n_pseudocells`: `[5]` 300 (unchanged), `[6]` 1350→**1700**.
- **CELL 12** (markdown): the `--ref-dir data/deconvolution/references/liver_GSE220075` path (×3) is
  STILL CORRECT (the clean ref was adopted in place). Add a lung example → `references/lung_native_pooled`.
  In the SLURM Job History table add the new production rows (LIVER 11170761 24.5 min, LUNG 11171098 14.0
  min) and mark the GSE178405-lung / Visium-liver rows superseded.
- **Re-execute only:** cells 1, 6, 9, 10, 13 (inventory/roster/fraction-histogram/checksum — refresh from
  the new outputs; lung's 34-type roster + new fractions appear automatically).

## pipeline9_analysis.ipynb (Stage 9 — tokenize + embed)
- **CELL 3** `min_len` dict: `"lung":277` → the new lung min token length from
  `logs/lung_native_casc_11171098.out` (tokenize_pseudocells.py stdout); `"liver":"?"` unchanged (or fill
  from `logs/liver_novis_casc_11170761.out`). Simplest robust option: set both to `"?"`.
- **CELL 12** (markdown) SLURM Job History: update LIVER (11170761) + LUNG (11171098) rows + a note that
  liver = Visium-excluded and lung = native pooled (replacing GSE178405).
- **Re-execute only:** cells 1,4,6,7,9,10,13 (tokenization/embedding/Augur/checksum figures).

## pipeline10_analysis.ipynb (Stage 10 — DE + positive controls)  ← biggest conceptual change
- **CELL 5** (markdown) — **now FALSE, must rewrite:** "18 of 178 blocks are hotspots. All are in BLOOD,
  SKMGN, SKMVL, HEART, and KIDNEY." → "**21 of 185 blocks are hotspots: blood 7, skmgn 5, skmvl 4, LUNG 3,
  heart 1, kidney 1. LUNG now shows exercise hotspots (native reference); liver still 0.**"
- **CELL 2** (markdown): "all 178 blocks" → "all 185 blocks".
- **CELL 6** (code): hardcoded `/178` in the suptitle f-string → use `{len(de)}` (or 185); ADD
  `"LIVER"` and `"LUNG"` keys to the `tissue_colors` dict (else a newly-hotspot lung block plots gray).
- **CELL 12** (markdown): refresh SLURM job rows (redetect 11173958) and re-derive the Tier-A recovery
  numbers + IHW-significant-gene ranges from the regenerated `pseudobulk_de/posctrl_summary.md`.
- **Re-execute only:** cells 1,3,4,8,9,10,13 (DE-count/posctrl figures + checksums).

## pipeline11_analysis.ipynb (Stage 11 — subspace probe)
- No study IDs / paths hardcoded. Update the global counts in **CELLS 0, 2, 12, 13**: "178 blocks" →
  "**185**"; "18"/"Eighteen" hotspots → "**21**"; the "(1 singleton block excluded)" note if the singleton
  count changed. **CELL 13** Reproducibility table → redetect job 11173958.
- **Re-execute only:** every code cell (all read the TSVs by tissue name; liver/lung rows + global counts
  refresh automatically).

## pipeline12_analysis.ipynb (Stage 12 — cross-species transfer)
- **CELL 3** `TRANSFER` dict + **CELL 2** markdown table: update the liver row (rat_genes 13134→12759,
  n_cells 300 unchanged) and lung row (rat_genes 14929→15348, n_cells 1350→1700) — pull the new
  `pct_ortho` / `eligible` / `count_mass` per tissue from `genecompass_input_human/transfer_comparison.md`.
  The CELL 3 plot title "total 8 950 across 10 tissues" → recompute (don't retype). **NB: lung's cell-type
  roster changed wholesale (native 34-type set), so every lung row changes identity, not just value.**
- **CELLS 0, 4, 6, 10** (markdown/prints): "16/18 PRESERVED … LOST 2/18" → "**PRESERVED 19 / WEAKENED 1 /
  LOST 1 of 21**"; re-derive the sex-axis medians + rat~human Spearman.
- **CELL 11**: SLURM history → transfer job 11174632; re-derive the ΔAUC range; the LOST/WEAKENED list now
  includes **lung / Pulmonary fibroblasts (WEAKENED, ΔAUC −0.15)**. Lung's other 2 hotspots are PRESERVED
  (Myeloid DC 0.833→0.85, Alveolar macrophages 0.823→0.83) — a new, reportable conserved-signal result.
- **Re-execute only:** cells 1,5,7,8,12 (read `transfer_comparison.tsv` dynamically).

---
**One-line summary for the reader of each notebook:** liver numbers barely move (Visium fix negligible);
lung changes throughout (native reference — new 34-type roster, new fractions, and **lung goes 0→3
exercise hotspots, 2 of which are conserved rat→human**); global counts 178→185 blocks, 18→21 hotspots.
