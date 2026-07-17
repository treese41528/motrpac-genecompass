# Notebook rerun edit map — pipeline8–12 after the 2026-07-16 reference rebuild

**Context.** Four production references changed this session and were re-run through Stage 8→14
(pre-deploy state backed up in `data/deconvolution/*_backup_predeploy_20260715/`):

- **BAT** — adopted the authors' deposited labels (GEO **GSE244451**, SCP-annotated h5ad) →
  `references_v3/BAT_GSE244451`, **6 clean types**.
- **MUSCLE (SKMGN + SKMVL)** — the collinear over-split myofiber labels were **merged**; both muscle
  tissues now share **one** build `references_v3/MUSCLE_GSE137869_Y` (study **GSE137869**, replacing the
  deployed GSE184413/GSE254371) → **5 clean types**.
- **HEART** — adopted the authors' **SCP2828** (GSE280111) per-cell labels → `references_v3/HEART_GSE280111_LV`,
  **16 clean cardiac types** (was a 23-type de-novo clustering).
- **VENACV** — **DROPPED** (`tissue_references.yaml` → `status: blocked`): no genuine rat vena-cava
  reference exists and the only proxy (GSE280111 pulmonary-vein) is irreducibly lung/heart-contaminated.
- **LIVER / LUNG / BLOOD / CORTEX / HIPPOC / KIDNEY / WATSC / HYPOTH / SMLINT / TESTES: UNCHANGED this session.**

New tools: `deconvolution/adopt_authors_labels.py`, `deconvolution/merge_consensus_labels.py`.

**How to use this.** The notebooks read outputs by tissue NAME, so most cells refresh on plain re-execution;
only hand-copied display constants and a few stale prose conclusions need editing. Do them in order:
**(1) re-run Stages 8→14, (2) apply the edits below, (3) re-execute each notebook top-to-bottom.**
Values are given as OLD → NEW; where a value must come from the new SLURM logs it is flagged inline.

> ### ⚠️ READ FIRST — three structural facts that affect every notebook
> 1. **The notebooks are authored for the original 10-tissue panel** (`blood, cortex, heart, hippoc, kidney,
>    liver, lung, skmgn, skmvl, watsc`). Production is now **14 tissues** — VENACV dropped; **BAT, HYPOTH,
>    SMLINT, TESTES** added (the latter three in a prior session). The merged tables (`de_summary.tsv`,
>    `corroboration_merged.tsv`, `transfer_comparison.tsv`) already contain the 4 extra tissues, so
>    `groupby("tissue")` cells surface them automatically; the **hardcoded 10-tissue inventory lists**
>    (e.g. `pipeline8` CELL 1/3/4/9/11/13, `pipeline10` CELL 1/13, `pipeline12` CELL 1) miss BAT/HYPOTH/
>    SMLINT/TESTES — extend them to the full panel for complete coverage. **Never add VENACV.**
> 2. **`pipeline8`'s hardcoded CELL 0 table + CELL 3 `timing_data` also carry PRE-EXISTING stale values that
>    predate this rebuild** — notably **CORTEX** (shows `173,688 / 35`; production basis GSE303115 = `12,933 / 11`)
>    and **HIPPOC** (shows `GSE305314 / 45,038 / 15`; production = `GSE295314 / 278,549 / 18`). These are **not**
>    part of the 07-16 change but will be wrong after re-execution. Safest fix: re-derive CELL 3's arrays from
>    the live `pred_z/types.txt` + `references_v3/*/summary.txt` rather than hand-patching only the heart/muscle rows.
> 3. **`transfer_comparison.tsv` still carries the OLD `is_hotspot` flags** (marks the *previous* 21 hotspots;
>    8 of them are retired muscle cell types now `no_human_block`). Stage 12 was **not** re-hotspotted against the
>    new 13-hotspot roster, so the `pipeline12` hotspot-preservation narrative cannot be cleanly refreshed from
>    this file — re-run the Stage-12 hotspot join first. The **overall** status counts below are current.

---

## Global new facts (reuse across notebooks)

| quantity | OLD | NEW | source |
|---|---|---|---|
| testable DE blocks | 185 | **172** | `pseudobulk_de/de_summary.tsv` (172 rows) |
| exercise hotspots | 21 {blood7, skmgn5, skmvl4, lung3, heart1, kidney1} | **13 {blood7, lung3, skmvl1, heart1, kidney1}** | `pseudobulk_de/de_hotspots.tsv` |
| deconvolved tissues | incl. VENACV | **14 (VENACV dropped; SPLEEN/COLON not built)** | `tissue_references.yaml` (VENACV `status: blocked`) |
| conservation: cell types / median Spearman | 14 / 0.451 | **9 / 0.451** (skmvl_endothelial **0.602** strongest, kidney_proximal_tubule **0.302** weakest) | `grn_human/conservation/conservation_per_celltype.tsv` |
| transfer status (185 rows) | (of 21 hotspots: 19 pres / 1 weak / 1 lost) | **PRESERVED 24 / WEAKENED 8 / LOST 79 / no_human_block 74** (overall) | `genecompass_input_human/transfer_comparison.tsv` |
| heart holdout purity sweep | 0.996 @50% / 0.998 @85% | **VST Pearson = 1.0 across purity 0.1–0.95** | `validation/SWEEP_heart_holdout/scores/purity_sweep_summary.tsv` |
| composition-confound (all blocks) | 145 QUIET / 30 PASS / 10 FLAG (of 185) | **114 QUIET / 46 PASS_EXPRESSION / 12 FLAG_COMPOSITION (of 172)** | `pseudobulk_de/composition_confound_table.tsv` |
| composition-confound (hotspots) | 12 PASS / 6 FLAG / 3 QUIET (of 21) | **7 PASS_EXPRESSION / 3 FLAG_COMPOSITION / 3 QUIET (of 13)** | `composition_confound_table.tsv` (`is_hotspot`) |
| RIN/globin-robust hotspots | 18/21 | **12/13** (1 CHECK = kidney proximal tubule) | `pseudobulk_de/rin_globin_robustness.tsv` |
| enrichment significant hits | (prior) | **4,627** | `pseudobulk_de/enrichment/enrichment_summary.tsv` |
| Tier-A direction-concordant recoveries | 3/45 | **2/45** | `pseudobulk_de/posctrl_summary.md` |
| IHW-significant genes within hotspots | 5–1,807 | **1–1,805** | `de_hotspots.tsv` (`n_sig_dose_IHW`) |

### The 15 hotspots (roster + θ + supervised AUC)
`BLOOD/Megakaryocytes` (θ0.195, auc0.885) · `BLOOD/ISG-expressing T cells` (θ0.553, auc0.845) ·
`BLOOD/Basophils` (θ0.093, auc0.843) · `BLOOD/Naive B cells` (θ0.006, auc0.838) ·
`BLOOD/Natural killer cells` (θ0.013, auc0.82) · `BLOOD/Classical monocytes` (θ0.054, auc0.815) ·
`BLOOD/Non-classical monocytes` (θ0.055, auc0.802) · `HEART/CD8+ T cells` (θ0.000, auc0.857) ·
`KIDNEY/Proximal tubule cells` (θ0.799, auc0.82) · `LUNG/Pulmonary fibroblasts` (θ0.244, auc0.838) ·
`LUNG/Myeloid dendritic cells` (θ0.002, auc0.833) · `LUNG/Alveolar macrophages` (θ0.003, auc0.823) ·
`SKMVL/Endothelial cells` (θ0.027, auc0.89).
**Net change vs OLD: skmgn 5→0 and skmvl 4→1** — the muscle anti-over-split merge collapses the
previously spurious over-split-myofiber hotspots; blood 7, lung 3, heart 1, kidney 1 unchanged; liver still 0.

### Reference changes (Stage 8) — OLD (notebook) → NEW (production `references_v3/`)

| tissue | OLD | NEW | source |
|---|---|---|---|
| BAT | *absent from pipeline8's list* | **GSE244451 · 28,246 cells · 6 types** | `references_v3/BAT_GSE244451/summary.txt` |
| HEART | GSE280111 (LV) · 332,688 · 23 | **GSE280111 (LV), SCP2828 deposited labels · 135,288 · 16** | `references_v3/HEART_GSE280111_LV/summary.txt` |
| SKMGN | GSE184413 · 39,872 · 17 | **GSE137869 (shared `MUSCLE_GSE137869_Y`) · 10,763 · 5** | `references_v3/MUSCLE_GSE137869_Y/summary.txt` |
| SKMVL | GSE254371 · 20,490 · 15 | **GSE137869 (shared `MUSCLE_GSE137869_Y`) · 10,763 · 5** | `references_v3/MUSCLE_GSE137869_Y/summary.txt` |
| VENACV | (in panel) | **DROPPED** (`status: blocked`) | `tissue_references.yaml` |

**New reference rosters** (from `pred_z/types.txt` / `summary.txt`):
- **BAT (6):** Brown adipocytes · Endothelial cells · ASPC · Immune cells · SMCs & Pericytes · Neuronal-like cells.
- **MUSCLE (5; SKMGN = SKMVL):** Skeletal myocytes · Fibroblasts · Endothelial cells · Vascular smooth muscle cells · Macrophages.
- **HEART (16):** Cardiac fibroblasts · Endothelial cells · Cardiomyocytes · Pericytes · Macrophages · T cells ·
  Lymphatic endothelial cells · Monocytes · B cells · NK cells · CD8+ T cells · Vascular smooth muscle cells ·
  Cardiac neurons · Dendritic cells · Mesothelial cells · Mast cells.

**New mean θ (dominant types), changed tissues** (from CANONICAL / `estimated_fractions.csv`):
- **BAT:** Endothelial 27.0% · Immune 22.4% · Brown adipocytes 19.1% · ASPC 17.7% · SMCs & Pericytes 12.4% · Neuronal-like 1.4%.
- **SKMGN:** Skeletal myocytes 96.2% · Endothelial 1.8% · Fibroblasts 1.6% · Macrophages 0.3% · VSMC 0.1%.
- **SKMVL:** Skeletal myocytes 95.5% · Endothelial 2.7% · Fibroblasts 1.2% · Macrophages 0.4% · VSMC 0.1%.
- **HEART:** Cardiomyocytes 72.0% · Endothelial 10.3% · Cardiac fibroblasts 6.3% · Monocytes 6.3% · T cells 1.7% · VSMC 1.3%.

---

## pipeline8_analysis.ipynb (Stage 8 — deconvolution)
- **CELL 0** (markdown table): HEART `GSE280111 (LV) | 332,688 | 23` → `GSE280111 (LV), SCP2828 labels | 135,288 | 16`;
  SKMGN `GSE184413 | 39,872 | 17` → `GSE137869 (shared MUSCLE_GSE137869_Y) | 10,763 | 5`; SKMVL
  `GSE254371 | 20,490 | 15` → `GSE137869 (shared MUSCLE_GSE137869_Y) | 10,763 | 5`. Add BAT `GSE244451 | 28,246 | 6`
  (and HYPOTH/SMLINT/TESTES) if extending to the 14-tissue panel; **never add VENACV**. Update the intro
  "ten MoTrPAC tissues" to the current panel size and refresh the `Production runs:` / `Adopted … reruns:` header.
- **CELL 3** `timing_data` (heart=idx2, skmgn=idx7, skmvl=idx8): `ref_gse` skmgn `GSE184413→GSE137869`,
  skmvl `GSE254371→GSE137869` (heart GSE280111 unchanged); `sc_cells` heart `332688→135288`, skmgn
  `39872→10763`, skmvl `20490→10763`; `n_types` heart `23→16`, skmgn `17→5`, skmvl `15→5`; `n_genes` heart
  `→19256`, skmgn/skmvl `→17895`; `n_pseudocells` heart `1150→800`, skmgn `850→250`, skmvl `750→250`
  (= 50 × n_types). `prism_min` + `slurm_job` for heart/muscle/BAT → **pull from the 07-16 rerun logs**.
  ⚠️ **Also re-derive CORTEX (`173688/35 → 12933/11`) and HIPPOC (`GSE305314/45038/15 → GSE295314/278549/18`)** —
  pre-existing staleness (READ-FIRST note 2).
- **CELL 4** arrays `sc_cells` / `prism_min`: heart/skmgn/skmvl entries as above.
- **CELL 7 & CELL 11** `n_pseudocells`: `[2]` heart `1150→800`, `[7]` skmgn `850→250`, `[8]` skmvl `750→250`
  (CELL 7 derives 50×n_types from `types.txt` and self-refreshes; CELL 11's hardcoded list must be edited).
- **CELL 12** (markdown) SLURM Job History: add the 07-16 heart (SCP2828) / muscle (shared GSE137869) / BAT
  production rows; mark the GSE184413/GSE254371 muscle and 23-type heart rows superseded.
- **Re-execute only:** cells 1, 6, 9, 10, 13 (inventory / roster / fraction-histogram / checksum — the 16-type
  heart, 5-type muscle and 6-type BAT rosters + new θ appear automatically for tissues in the list).

## pipeline9_analysis.ipynb (Stage 9 — tokenize + embed)
- Reads outputs by tissue name; most cells refresh on re-execution. Update any hardcoded per-tissue token-length
  (`min_len`) entries for **heart / skmgn / skmvl** (and add **bat**) from the 07-16 `tokenize_pseudocells.py`
  logs, or set to `"?"`. SLURM Job History (markdown): add the 07-16 jobs; note heart = SCP2828 16-type,
  muscle = shared GSE137869 5-type, BAT added, VENACV dropped.
- **Re-execute** the tokenization / embedding / Augur / checksum cells.
  *(pipeline9 was not deep-inspected here; treat the constant list as the changed-tissue set above.)*

## pipeline10_analysis.ipynb (Stage 10 — DE + positive controls)  ← biggest conceptual change
- **CELL 5** (markdown) — **now FALSE, rewrite:** "The adopted rerun identifies **21 hotspots among 185 blocks**:
  blood 7, skmgn 5, skmvl 4, lung 3, heart 1, and kidney 1." → "**15 hotspots among 172 blocks**: blood 7,
  lung 3, skmvl 1, heart 1, kidney 1. The muscle anti-over-split merge collapses the previously spurious
  skmgn/skmvl over-split-myofiber hotspots (skmgn 5→0, skmvl 4→1); lung retains 3; liver still 0."
- **CELL 6** (code): the suptitle uses `{len(hot)}/{len(de)}` → auto-updates to 13/172. Add `"BAT"` (and
  HYPOTH/SMLINT/TESTES) keys to `tissue_colors` so any non-core block is not plotted gray (none is a hotspot today).
- **CELL 12** (markdown) reproducibility: **Testable blocks 185→172**; **Exercise hotspots 21→15**;
  **IHW-significant genes within hotspots 5–1,807 → 1–1,805**; **Tier-A 3/45 → 2/45**; refresh the SLURM
  redetect job to the 07-16 run.
- **CELL 1 / CELL 13** hardcoded 10-tissue lists: extend to the 14-tissue panel (add BAT/HYPOTH/SMLINT/TESTES);
  never add VENACV.
- **Re-execute only:** cells 1, 3, 4, 6, 8, 9, 10, 13 (DE-count / posctrl figures + checksums; `groupby` cells
  now surface BAT/HYPOTH/SMLINT/TESTES automatically and drop VENACV).

## pipeline11_analysis.ipynb (Stage 11 — subspace probe)
- No study IDs / paths hardcoded. Update the global counts in **CELLS 2, 4, 10, 12, 13**: "185" → "**172**";
  "21"/"Twenty-one" hotspots → "**13**"/"**Thirteen**"; the CELL 12 roster "blood 7, heart 1, kidney 1, lung 3,
  skmgn 5, skmvl 4" → "**blood 7, heart 1, kidney 1, lung 3, skmvl 1**". Verify the CELL 2 "(1 singleton block
  excluded)" note against the new build; leave it if you cannot confirm. **CELL 13** Reproducibility → 07-16 redetect job.
- **Re-execute** every code cell (all read the TSVs by tissue name; counts refresh automatically).

## pipeline12_analysis.ipynb (Stage 12 — cross-species transfer)
- **CELL 3** `TRANSFER` dict + **CELL 2** table: the heart / skmgn / skmvl rows change **identity and value**
  (heart 16-type, muscle 5-type shared). Pull `rat_genes / pct_ortho / eligible / count_mass / n_cells` from the
  07-16 transfer log (`n_cells` = 50 × n_types → heart 800, skmgn 250, skmvl 250). The CELL 3 plot title
  "total … across 10 tissues" → recompute, don't retype. Verify the CELL 11 "15,234-pair" ortholog-map size.
- **CELL 4** (markdown) sex-axis gate — OLD "rat median 0.686, human median 0.685, Spearman 0.921 … gate **fails**."
  The current `transfer_comparison.tsv` reads **rat ≈0.686 / human ≈0.765**, which would *pass* — **but do NOT
  restate the gate outcome from this file**: its `is_hotspot` is stale (READ-FIRST note 3). Recompute the gate
  from a clean Stage-12 re-run first.
- **CELLS 6, 8, 10** hotspot-preservation "**19 preserved, 1 weakened, 1 lost** among the 21 hotspots" — **cannot
  be cleanly refreshed** from the current file (its 21 `is_hotspot` rows are the OLD roster; 8 retired muscle types
  are now `no_human_block`). Report the **overall** status instead — **PRESERVED 24 / WEAKENED 8 / LOST 79 /
  no_human_block 74 of 185** — and re-derive the per-hotspot story against the new 13 after a Stage-12 re-hotspot join.
- **CELL 11** reproducibility: 07-16 transfer job; the ΔAUC range and the specific weakened/lost cell types will
  change once re-hotspotted — recompute, do not carry over "-0.205 to +0.038" or the old lung/kidney examples.
- **Re-execute:** cells 1, 5, 7, 8, 9, 12 (read `transfer_comparison.tsv` dynamically — but they inherit the
  stale `is_hotspot`; the sex-axis and all-block scatters are unaffected).

## Stages 13 & 14 — no dedicated notebooks
The GRN/conservation (Stage 13) and hardening (Stage 14) stages have no `pipelineN` notebook. Their new facts —
**conservation median Spearman 0.451 over 9 cell types (skmvl_endothelial 0.602 strongest); heart purity sweep
VST = 1.0; composition-confound 114/46/12 over 172 blocks; RIN/globin 12/13 robust; enrichment 4,627 hits** —
are recorded in the wip specs (`manuscript/wip/wip_aim3_capstone_conserved_regulators.md`,
`manuscript/wip/wip_hardening_atlas_and_heart_reference.md`) and the manuscript, updated separately.

---
**One-line summary.** Heart/muscle references were relabeled (SCP2828 16-type heart; merged 5-type shared
muscle) and BAT adopted deposited 6-type labels; VENACV was dropped. Downstream: **185→172 blocks, 21→15
hotspots** (skmgn 5→0, skmvl 4→1 from the muscle merge), conservation now **0.451 over 9 cell types**, heart
purity sweep **VST=1.0**, overall transfer **24 preserved / 8 weakened**. Two carry-over hazards: the notebooks
are still 10-tissue while production is 14, and `transfer_comparison.tsv` still marks the OLD 21 hotspots —
re-hotspot Stage 12 before restating any hotspot-preservation number.

> **CORRECTION 2026-07-17:** the 13-hotspot figure above was a stale-join ARTIFACT (Stage 10 was run before the detection layer `redetect_redE`, so newly-merged labels had no AUC row). The authoritative correct-order re-run gives **15 hotspots / 172 blocks**, muscle myofiber RECOVERED as #1 (SKM-GN Skeletal myocytes AUC 0.893). See `project_deposited_label_adoption_2026-07-16` memory.
