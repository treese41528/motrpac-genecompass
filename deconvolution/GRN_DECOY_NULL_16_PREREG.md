# GRN decoy null on the 16-tissue default (pre-registered 2026-11-04, before any decoy network was built)

**Why.** The decoy null (`GRN_DECOY_NULL_PREREG.md`, result `GRN_DECOY_NULL_RESULTS.md`) was run on the eight
networks of the refs4 (14-tissue) run. The default is now the 16-tissue clean rebuild, with six networks
(`deconvolution.grn_dir` = `data/deconvolution/grn_v2_refs4all_rebuild/`). Those six are, edge for edge, six of
the eight refs4 networks (`ALL_TISSUE_RERUN_RESULTS.md`, `REFS4ALL_REBUILD_RESULTS.md`), but the null was never
built from the default's own inputs. This reruns it there.

**Scope.** The six rat-space networks of the default: BLOOD classical monocytes, erythroid cells,
megakaryocytes and platelets, neutrophils; LUNG vascular smooth muscle cells; SKMGN tenocytes. Inputs: the
default's `genecompass_input_v2_refs4all_rebuild/<tissue>/dataset` and `augur_input_v2_refs4all_rebuild/<tissue>/meta.tsv`;
the v2 checkpoint. Arguments exactly as Stage 13 and the first null: `--bootstrap 30 --seed 0 --decoy-seed 0`.
Decoys go to `data/deconvolution/grn_v2_refs4all_rebuild_decoy/`; scored by `deconvolution/grn_decoy_fdr.py`
(defaults: tf-dir = the config's `grn_dir`).

**Readouts and reading.** Unchanged from `GRN_DECOY_NULL_PREREG.md` (edge level: pooled and per-network FDR(2)
with the ≤ 0.10 / ≤ 0.50 / > 0.50 reading and t*; hub level: each TF's Σ|Δ| against its network's decoys, BH
within network, q < 0.10, anchors Mef2c/Nr4a1/Myf6/Tfeb/Epas1 where present). The verdict is read over the six
default networks.

**Reproduction check (reported, not a verdict).** For each of the six networks, the decoy table is compared with
the refs4 decoy table of the same name (`grn_v2_refs4_decoy/`), joined on (tf, target): same edge set and
max |Δ delta|, |Δ boot_sd|, |Δ z|. Identical inputs and seeds should give identical decoys; any difference is
reported with its cause.

No other threshold, network subset or matching rule will be tried after the results are seen.
