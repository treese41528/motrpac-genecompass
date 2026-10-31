# GRN decoy null: result (2026-10-31)

Pre-registration: `GRN_DECOY_NULL_PREREG.md`, committed before any decoy network was built. Job 12103791
(1 h 24 min, A100). Decoy networks: `data/deconvolution/grn_v2_refs4_decoy/`. Record tables (copied from
`<decoy dir>/fdr/`): `grn_decoy_record/` (`summary.txt`, `fdr_per_network.tsv`, `fdr_curve_pooled.tsv`,
`hub_vs_decoy.tsv`). Scorer: `grn_decoy_fdr.py`.

**Question.** Is the GRN's `|z| ≥ 2` "confident edge" an error rate, or only a ranking? And does any TF's
hub strength exceed that of a random gene of the same expression?

**Verdict, edge level: FDR(2) = 1.03, so `|z| ≥ 2` edges are not distinguishable from random-gene
deletions.** By the registered reading they are no longer called "confident" anywhere in the docs or code
output (done in the same commit).

| Network | TFs (= decoys) | Fraction of TF edges with \|z\| ≥ 2 | Same, decoy edges | FDR(2) |
|---|---|---|---|---|
| BLOOD Classical monocytes | 105 | 0.145 | 0.149 | 1.03 |
| BLOOD Erythroid cells | 123 | 0.119 | 0.124 | 1.04 |
| BLOOD Megakaryocytes and platelets | 110 | 0.140 | 0.142 | 1.01 |
| BLOOD Neutrophils | 117 | 0.106 | 0.103 | 0.97 |
| LUNG Endothelial cells | 140 | 0.089 | 0.089 | 1.00 |
| LUNG Mesothelial cells | 118 | 0.083 | 0.090 | 1.08 |
| LUNG Vascular smooth muscle cells | 149 | 0.074 | 0.078 | 1.04 |
| SKMGN Tenocytes | 119 | 0.094 | 0.101 | 1.08 |
| **Pooled** | | 215,429 edges | 221,796 edges | **1.03** |

The decoy fraction matches the TF fraction at every threshold, not only at 2: pooled FDR is 1.02–1.06 from
t = 2 to t = 5, and no t on the grid 2–50 reaches 0.10 (t* does not exist).

**Verdict, hub level: no TF in any network beats its expression-matched decoys.** 0 of 981 TFs reach
q < 0.10 (BH within network). In five networks the top TF sits at the p floor that 105–149 decoys allow
(about 0.007–0.008), which is what the largest of about 120 draws from the null does; the best q in any
network is 0.37 (LUNG mesothelial) and most are 0.84–0.96. Every anchor that is present fails: Nr4a1 (classical monocytes)
q = 0.96, Tfeb (erythroid, tenocytes) q = 0.97–0.98, Epas1 (all three LUNG networks, tenocytes)
q = 0.90–0.97. Mef2c and Myf6 are not expressed TFs in any of these eight networks.

**Reading.**
- Deleting a TF's token moves the embedding no more, and no more differently between the trained and control
  arms, than deleting a random gene of the same expression. In this model and design the differential GRN
  carries no TF-specific signal: neither its edges nor its hub ranking.
- This agrees with, and sharpens, `RETRAIN_V2_RESULTS.md` §4, where hub strength followed the TF's own
  expression (Spearman about 0.8). The decoy result says expression is essentially all of it.
- The null is conservative by construction (some non-TF genes have real effects, which can only fatten the
  decoy tail), but a conservative FDR of 1.03 still leaves no room for a TF-specific excess.
- What it does not touch. The perturbation engine's own pre-registration (Module A: Mef2c 98th percentile,
  target p = 2e-7) asked whether a deletion shifts its known targets, not whether a TF does so more than a
  random gene, and is not overturned here, but it no longer licenses reading the exercise GRN as
  TF-specific regulation. The rat-human "conservation" was already model self-consistency, not
  cross-species biology; it is now self-consistency of an expression-driven quantity.
- Consequence for the docs. Aim 2b's regulator lists and Aim 3b's "conserved regulators" are reported as
  expression-driven model readouts, not as regulators. Hub tables stay in the record as descriptive
  outputs; no hub is named as an exercise regulator on GRN evidence.

No other threshold, network subset or matching rule was tried. A second decoy seed was not run.
