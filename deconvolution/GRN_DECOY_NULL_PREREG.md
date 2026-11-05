# A decoy null for the GRN's |z| (pre-registered 2026-10-31, before any decoy network was built)

**Question.** The differential GRN calls an edge "confident" when |z| ≥ 2 (z = trained−control target
shift over its bootstrap SD). Is that threshold an error rate, or only a ranking? A label permutation
cannot answer it: control is week 0 by design, so any shuffle mixes doses into the control arm and the null
z is not distributed like the observed z (`GRN_NULL_WHY_ABANDONED.md`).

**The null.** Keep the labels, arms, cells, bootstrap and model exactly as they are and change only the gene
that is deleted. In each network, every expressed TF is replaced by a non-TF gene of the nearest pseudobulk
expression (mean of the two arm profiles, log1p distance, without replacement, TFs matched in a seeded random
order; `build_grn_pooled.py --decoy-seed 0`). Expression is matched because hub strength follows a TF's own
expression (Spearman ~0.8, RETRAIN_V2_RESULTS.md §4). Under H0 "this TF's deletion acts like the deletion of a
random gene of the same expression", decoy and TF z are identically distributed, so

  FDR(t) = P_decoy(|z| ≥ t) / P_TF(|z| ≥ t)   (fractions of each network's candidate edges).

Some non-TF genes do have real effects, so the decoy tail is if anything heavier than a true null and the
FDR is conservative (over-stated). The null tests TF specificity, not "any effect of exercise at all".

**Scope.** The 8 rat-space networks of the default (refs4) run, `data/deconvolution/grn_v2_refs4/`, same
model, cells, `--bootstrap 30 --seed 0` as Stage 13. Decoys go to `data/deconvolution/grn_v2_refs4_decoy/`.
Scored by `deconvolution/grn_decoy_fdr.py`.

**Readouts and how they will be read (fixed now).**

1. *Edge level.* Pooled FDR(2) = (decoy edges with |z| ≥ 2 / all decoy edges) / (TF edges with |z| ≥ 2 / all
   TF edges), over the 8 networks; the same per network.
   - FDR(2) ≤ 0.10 → |z| ≥ 2 edges carry an FDR of at most 10% against random-gene deletion; report it.
   - 0.10 < FDR(2) ≤ 0.50 → |z| stays a ranking. Report FDR(2) and t*, the smallest t on the grid 2, 2.5, …, 50
     with pooled FDR ≤ 0.10 and at least 100 TF edges; if none, say so.
   - FDR(2) > 0.50 → |z| ≥ 2 edges are not distinguishable from random-gene deletions; stop calling them
     "confident" anywhere in the docs.
2. *Hub level.* Per network, each TF's hub strength Σ|Δ| over all its edges against the decoys' Σ|Δ| in the
   same network: p = (1 + #decoys ≥ TF) / (1 + #decoys), BH within network. Report how many TFs reach q < 0.10
   per network and whether the named anchors (Mef2c, Nr4a1, Myf6, Tfeb, Epas1, wherever present) do.
   The hub result does not depend on the edge result.

No other threshold, network subset or matching rule will be tried after the results are seen. A second decoy
seed may be run only to report seed variability, and would be reported beside seed 0, not instead of it.
