# Perturbation-engine pre-registration (Module A / A.6)

**Frozen 2026-07-03, BEFORE the first real perturbation run.** Mirrors `POSCTRL_PREREG.md` (the DE one):
commit the positive controls, expected directions, acceptance criteria, and miss-reading ladder in
advance, so a passing result is a genuine test and a failure is diagnosed in a fixed order rather than
explained away. Gene set is frozen in `reference/perturb_prereg_genes.tsv` (all rel-113 ENSRNOG,
verified in-vocab). Engine: `finetune/genecompass/perturb_cells.py`.

## What the engine outputs (definitions)
- **cell shift** = `1 - cos(CLS_original, CLS_perturbed)`, mean over the cells of a (tissue × cell type)
  that express the gene. Larger = the deletion moves the cell's representation more = bigger effect.
- **predicted targets** = the surviving genes whose *contextual* embeddings shift most when the gene is
  deleted (`1 - cos` per gene position, original-minus-deleted aligned to perturbed). These are the
  candidate GRN edges Module B consumes.

## Positive controls (pre-registered, directional)
1. **Mef2c in skeletal muscle** (`Mef2c` `ENSRNOG00000033134`, cell_type "Skeletal muscle", SKMVL/SKMGN).
   Mef2c is a myogenic master-TF. **Expected:** (a) its cell shift is in the **top decile** of a null of
   ~50 random genes *matched on mean expression value/rank* in that cell type (deletion of a hub TF should
   move the cell more than a typical equally-expressed gene); (b) its **top-25 predicted targets are
   enriched** for the frozen Mef2/myogenic set (`Myog, Ckm, Des, Actn2`, + coherent muscle-structural
   genes) above chance (hypergeometric p < 0.05 against the cell's expressed-gene background).
2. **Spi1 (PU.1) / Cebpa in blood monocytes** (`ENSRNOG00000012172` / `ENSRNOG00000010918`, cell_type
   "Classical monocytes"). Myeloid master-TFs. **Expected:** cell shift in the top decile of the matched
   null; targets enrich for myeloid genes. (Cross-lineage replicate of control 1 — guards against a
   muscle-only artifact.)

## Sanity / negative controls
- **Not-expressed gene → ~zero shift.** `Alb` (hepatocyte) and `Snap25` (neuronal) deleted in skeletal
  muscle: the engine should find them in ~0 cells (`n_cells≈0`) or, where trace-present, a shift below the
  matched-null median. If a not-expressed gene produces a large shift, the token map or the delete is wrong.
- **Self-consistency:** across a panel of perturbed genes in a cell type, Spearman(cell shift, gene mean
  value) > 0 — higher-expressed genes carry more of the profile, so deleting them shifts the cell more.

## Acceptance criteria (all four)
| # | criterion | pass threshold |
|---|---|---|
| 1 | Mef2c cell shift vs matched-expression null | > 90th percentile (one-sided) |
| 2 | Mef2c top-25 targets vs frozen Mef2/myogenic set | hypergeometric p < 0.05 |
| 3 | Not-expressed controls (Alb, Snap25) in muscle | n_cells≈0 or shift < null median |
| 4 | Self-consistency Spearman(shift, mean value) | ρ > 0 |
Myeloid control (Spi1/Cebpa) is a **replication** target, not a gate — report it, don't fail on it alone.

## Miss-reading ladder (diagnose a failing control IN THIS ORDER before concluding the engine is broken)
1. **Coverage** — is the gene's token actually present in the cells? (`gene ∈ rat_tokens.pickle`, and the
   token appears in `input_ids`.) A missing token → no perturbation, not a null result.
2. **Power** — enough cells expressing it? (`n_cells`; a cell type with <~20 expressing cells is
   under-powered, not negative.)
3. **Implementation** — does delete actually change the embedding? (CPU unit test already passes;
   confirm original ≠ perturbed CLS on one cell.)
4. **Biology** — is the expected target set right *for rat*? (The Mef2 targets are ortholog-reasoned;
   a real miss here is informative, not a bug.) Only after 1–3 clear do we treat a control miss as a
   negative biological result.

## Confounds to hold in view (carry from the DE reporting policy)
- The **dominant parenchyma** (skeletal muscle here) is the mRNA-bias driver for its tissue
  ([[downstream-bias-claim-guidance]]); perturbation is on the *embedding* (target-sum-normalized, bias-
  insulated), but interpret target *magnitudes* relatively, not absolutely.
- GeneCompass has **no cross-cell attention** → this is a per-cell mechanistic readout, not a population
  GRN; Module B aggregates across cells to build the differential trained-vs-control network.
