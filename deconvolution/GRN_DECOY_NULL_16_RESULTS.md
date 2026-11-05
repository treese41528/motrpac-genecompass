# GRN decoy null on the 16-tissue default: result (2026-11-05)

Pre-registration: `GRN_DECOY_NULL_16_PREREG.md` (780b149), committed before any decoy network was built.
Job 12192357 (57 min, A100). Decoys: `data/deconvolution/grn_v2_refs4all_rebuild_decoy/`. Record tables:
`grn_decoy_record_16/`. Scorer: `grn_decoy_fdr.py`, run with its defaults (the config's `grn_dir`).

**Verdict, edge level: FDR(2) = 1.03.** By the registered reading, `|z| ≥ 2` edges cannot be told apart from
random-gene deletions. This is the same verdict as the first run.

| Network | TFs (= decoys) | TF edges with \|z\| ≥ 2 | Decoy edges with \|z\| ≥ 2 | FDR(2) |
|---|---|---|---|---|
| BLOOD Classical monocytes | 105 | 0.145 | 0.149 | 1.03 |
| BLOOD Erythroid cells | 123 | 0.119 | 0.124 | 1.04 |
| BLOOD Megakaryocytes and platelets | 110 | 0.140 | 0.142 | 1.01 |
| BLOOD Neutrophils | 117 | 0.106 | 0.103 | 0.97 |
| LUNG Vascular smooth muscle cells | 149 | 0.074 | 0.078 | 1.04 |
| SKMGN Tenocytes | 119 | 0.094 | 0.101 | 1.08 |
| **Pooled** | | 169,219 edges | 173,828 edges | **1.03** |

Pooled FDR stays between 1.00 and 1.05 from t = 2 to t = 7. No t on the grid 2–50 reaches 0.10, so t* does not
exist.

**Verdict, hub level: no TF beats its expression-matched decoys.**
- 0 of 723 TFs reach q < 0.10 (BH within network). The best q in any network is 0.84 (SKMGN tenocytes).
- Every anchor that is present fails:
  - Nr4a1 (classical monocytes): q = 0.96.
  - Tfeb (erythroid cells, tenocytes): q = 0.97–0.98.
  - Epas1 (LUNG vascular smooth muscle, tenocytes): q = 0.90–0.92.

**Reproduction check.** All six decoy tables are identical to the refs4 decoy tables of the same name: same edge
sets (223,105–309,899 edges each), with max |Δ| of 0 in delta, boot_sd and z. The per-network FDR(2) values
therefore equal the first run's for these six networks. Identical inputs and seeds gave identical decoys.

**Reading.** On the default's own networks, as in the first run, the differential GRN carries no TF-specific
signal, in its edges or in its hub ranking. The consequences already drawn in `GRN_DECOY_NULL_RESULTS.md` stand
unchanged: no "confident" edges, and no hub is named as a regulator on GRN evidence. No other threshold, network
subset or matching rule was tried.
