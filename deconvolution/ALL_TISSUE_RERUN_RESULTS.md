# All-tissue (16) rerun: result (2026-11-01)

Registration: `REFERENCE_RELABEL_PREREG.md` §14 "What follows" and the §16 entry of 2026-11-01 (119e450), both
committed before any of this was run. Profile: `config/profiles/refs_v4all.yaml` (now the default config, as
§14 registered; the 14-tissue refs4 result is `config/profiles/refs4_14.yaml`). Jobs: gate 12108627 (6 h),
Stage 10 12108628, Stage 12 12108629, follow-up 12108630 (Stage 13, Stage 14). The 14 refs4 tissues were
linked in; the file listing of every linked target is identical before and after (nothing written through
the links).

**What changed.** Two tissues joined: SPLEEN (GSE306764, 14 types) and OVARY (GSE343214, 18 types; females
only, 24 samples). 197 blocks (was 165), 151 analysable under the §6 rule (was 124). Both new tissues are
read differentially only (SPLEEN fractions are implausible; OVARY is one animal of unknown strain).

## Headline, 16 tissues vs refs4 (14)

| | refs4 (14) | All tissues (16) |
|---|---|---|
| Blocks / analysable (§6) | 165 / 124 | 197 / 151 |
| Exercise hotspots (gate q < 0.05) | 11 | **9** |
| Dose-significant genes (IHW < 0.05, analysable blocks) | 48,522 | 53,084 (SPLEEN 6,439) |
| Augur–PLS Spearman (analysable, both scored) | 0.788 (118) | 0.785 (130) |
| Embedding beats PCA (analysable) | 71/124, +0.019, p 0.013 | 82/151, +0.009, **p 0.15** |
| Median trained AUC (max) | 0.603 (0.887) | 0.605 (0.887) |
| Transfer: hotspots preserved | 11/11 | 9/9 |
| Transfer, all blocks: preserved / weakened / lost | 42 / 15 / 107 | 44 / 29 / 123 |
| Transfer sex check (median AUC rat → human; Spearman) | PASS | PASS (0.743 → 0.745; 0.930) |
| RIN/globin: hotspots robust | 11/11 | 9/9 |
| GRNs; self-consistency median | 8; 0.310 (859 TFs) | 6; 0.310 (633 TFs) |
| Positive controls, Tier A / Ai / B | 5/45, 2/14, 7/22 | 6/45, 2/14, 7/22 |

## refs4 hotspots that cross q = 0.05 (registered report, both q)

| Hotspot | q, refs4 | q, all tissues | AUC |
|---|---|---|---|
| LUNG Endothelial cells | 0.0440 | 0.0535 | 0.786 |
| LUNG Mesothelial cells | 0.0440 | 0.0535 | 0.785 |

Their AUCs and p values are unchanged: the gate is one BH family, and the 32 new blocks (none significant)
enlarge it. The other nine keep their status; every q rises by the same factor (about 1.2). No new hotspot.

## SPLEEN and OVARY

- **Gate.** No SPLEEN or OVARY block reaches q < 0.05. Best: OVARY smooth muscle (AUC 0.845, p 0.012,
  q 0.081), macrophages, pericytes and pDC (q 0.089–0.094). SPLEEN's best is red-pulp macrophages (AUC 0.663,
  q 0.26). OVARY has 24 samples (5 controls per sex arm, one sex), so its p values have a coarse floor.
- **Augur.** OVARY has no Augur score: with one sex, each block has 5 control pseudo-cells, below the 10 per
  arm Augur subsamples. TESTES is excluded the same way. The Augur–PLS correlation is over the other blocks.
- **DE.** SPLEEN: 6,439 dose-significant genes, spread evenly over all 12 analysable types (461–745 each;
  500 of the 1,489 genes are significant in at least half the blocks). The same sharing is seen in BLOOD and
  LUNG; it is the shared bulk signal of deconvolved profiles, not a cell-type-specific response. SPLEEN has
  DE but no embedding signal.
- **OVARY has 0 dose-significant genes, by construction.** `run_pseudobulk_de.R` combines the two sexes'
  training p values by Fisher's method and sets the combined p to missing when either sex is absent
  (line 255). A single-sex tissue therefore never has a dose p value. **This also explains TESTES's 0 genes
  in every earlier run**: it is structural, not biology. Not changed here (it would change the registered
  method); see "Open".
- Five new blocks are skipped by §6 (OVARY CD4+ T, erythroid, neutrophils; SPLEEN CD8+ T, monocytes).

## Other changes

- Tier A gains SKMGN Mfn1 (IHW 0.058 → 0.048) through the pooled refit; the same flip was seen in the
  HIPPOC-corrected arm (§13). No other control changes.
- The GRN step was rerun because the hotspot set changed; the six remaining networks are identical to refs4's
  (every edge and value equal after sorting rows; same per-network self-consistency). The two LUNG
  networks that dropped out stay in `grn_v2_refs4/` as refs4 outputs.
- Transfer: all nine hotspots preserved; the extra 14 "weakened" blocks are mostly new SPLEEN/OVARY blocks
  with near-chance rat AUCs.

## Reading

- Adding two tissues with no embedding signal costs two marginal LUNG hotspots (q 0.044 → 0.054) and takes
  the embedding-beats-PCA comparison below significance (p 0.013 → 0.15). Neither is a change in those
  blocks; both follow from the larger family. The nine surviving hotspots (BLOOD ×4, SKMVL ×2, SKMGN
  tenocytes, LUNG vascular smooth muscle, HYPOTH OPC) are unchanged in AUC, transfer and robustness.
- The registered default is now the all-tissue result. Any statement about LUNG endothelial or mesothelial
  cells as exercise hotspots must say "refs4 (14 tissues) only".

## Open

- **Single-sex DE.** TESTES and OVARY cannot yield dose-significant genes under the current Fisher combine.
  A fix (use the one sex's p for single-sex tissues) would change the registered method and the pooled
  IHW/repfdr fit for every tissue; it needs the user's decision and its own registration.
