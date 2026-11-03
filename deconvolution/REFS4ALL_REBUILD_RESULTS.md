# Clean rebuild of the 16-tissue default: results (2026-11-02)

Pre-registration: `REFS4ALL_REBUILD_PREREG.md` (bf0f417, written before any job ran). Commands: `DECONV_REBUILD_RUNBOOK.md`
§0.6–0.7 exactly as written, under `PIPELINE_PROFILE=config/profiles/refs4all_rebuild.yaml`, at commit aee68a3.

- **Per-tissue jobs:** 16 Stage 8–9 jobs, `deconv_<tissue>`, 12121714–12121729, all COMPLETED.
- **Cross-tissue job:** 12122211, Stages 11 → 10 → 12 → 13 → 14, COMPLETED in 20 h 26 min.
- **Comparison:** `python deconvolution/compare_arms.py --b config/profiles/refs4all_rebuild.yaml --out tmp/rebuild_compare`
  (job 12134899). It pairs every output file of the two arms by path.

**Verdict: the rebuild REPRODUCES the default.** Every pre-registered layer is IDENTICAL, apart from three things:
- **GRN row order:** the GRN tables hold the same edges and values, but tied rows are written in a different
  order. Cause and fix below.
- **Stage 14 comparability:** only 3 sweeps can be compared, because the default tree holds older Stage 14
  sweeps built from earlier references.
- **BAT and HEART sweep r:** equal to 8 decimals, not 9. The old sweep files rounded the truth fractions to 7 digits.

## Scored against the pre-registration

| Layer | Pre-registered IDENTICAL | Rebuild | Verdict |
|---|---|---|---|
| Stage 8 θ | max abs diff ≤ 1e-9, 16 tissues | 0.0 in all 16; `estimated_fractions.csv` byte-identical | IDENTICAL |
| Stage 8 Z | max rel diff ≤ 1e-9 | all 261 compared files in `results_dir` byte-identical (θ, `pred_z/`; `.rds` skipped) | IDENTICAL |
| Stage 9 embeddings | equal | every `cell_embeddings.npy` byte-identical | IDENTICAL |
| Stage 11 gate | 197 / 151 / same 9 hotspots, AUCs within 1e-9 | 197 / 151 / the same 9; max diff 0.0 in every numeric column of `corroboration_merged.tsv` | IDENTICAL |
| Stage 10 DE | 53,084 dose genes, same genes | every DE table byte-identical | IDENTICAL |
| Stage 12 transfer | 9/9 | all 127 files in `genecompass_input_v2_refs4all_human/` byte-identical | IDENTICAL |
| Stage 13 GRNs | 6, edges equal | 6 rat + 6 human; same edge sets, numeric diff 0.0 on every edge; rows of tied \|z\| in a different order | IDENTICAL (content) |
| Stage 14 purity r | equal | BAT / HEART / HYPOTH: equal (max \|Δr\| 9e-9); the other 8 sweeps have no current default to compare against (below) | IDENTICAL where comparable |

File counts (`compare_summary.tsv`):

| Folder | IDENTICAL | DIFFERS | ONLY in rebuild |
|---|---|---|---|
| `results_refs4all` | 261 | 0 | 0 |
| `genecompass_input_v2_refs4all` | 788 | 4 | 4 |
| `genecompass_input_v2_refs4all_human` | 127 | 0 | 4 |
| `augur_input_v2_refs4all` | 48 | 0 | 0 |
| `grn_v2_refs4all` | 0 | 6 | 0 |
| `grn_v2_refs4all_human` | 3 | 6 | 0 |

Every non-identical file is explained:

- **`pseudobulk_de*/de_methods.tsv`** (3 files): the rebuild has an extra `seed 1` row. Recording the IHW seed
  was added after the default was run, and the seed in use was the same.
- **`pseudobulk_de/posctrl_summary.md`**: the header prints its own folder path.
- **`summary.txt`** for BAT / HEART / HYPOTH / TESTES (only in the rebuild): the default links these four tissues
  from July runs, which did not write this file.
- **GRN `*_pooled.tsv`** (6 rat, 6 human):
  - What matches: joined on (tf, target), every edge is present in both arms with identical delta, boot_sd and
    z (max diff 0).
  - What differs: only the order of rows with tied |z|.
  - Cause: edges are sorted by |z| alone, so tied rows keep the order of their candidate list. That list
    comes from iterating a set of gene-ID strings, and Python randomises string hashing per process.
  - Fixed in `finetune/genecompass/build_grn_pooled.py` and `build_grn_human.py`: ties are now broken by
    (tf, target). The candidate ranking gets the same tie-break, so a `--top-edges` cut cannot depend on the
    hash seed either. The default's cap is 0 (no cut), so no edge set was ever affected.

## Stage 14: what the default tree holds

The default `validation_dir` (`data/deconvolution/validation/`) holds sweeps built from older references. The
runbook (§0.7) notes that Stage 14 refuses to reuse them: a sweep without a matching `sweep_spec.json` is
stale. The rebuild is therefore the first run of the purity sweeps on the current production references.

| Sweep | Rebuild reference | Default sweep built | Comparable? | Rebuild r at 10 / 50 / 85 / 95% purity | max \|Δr\| vs default |
|---|---|---|---|---|---|
| bat | `references_v3/BAT_GSE244451` | 07-27, same reference | yes | 0.988 / 0.998 / 0.999 / 1.000 | 9e-9 |
| heart | `references_v3/HEART_GSE280111_LV` | 10-05, same reference | yes | 0.982 / 0.998 / 1.000 / 1.000 | 8e-9 |
| hypoth | `references_v3/HYPOTH_GSE248413_Y` | 10-05, same reference | yes | 0.955 / 0.992 / 0.998 / 0.999 | 0 |
| testes | `references_v3/TESTES_OMIX767_lifted` | 07-27, raw-ID reference (before the 10-22 lift) | no | 0.998 / 1.000 / 1.000 / 1.000 | 4e-5 |
| cortex | `references_v4/CORTEX_GSE303115` | 07-06, pre-relabel | no | 0.997 / 1.000 / 1.000 / 1.000 | 0.003 |
| hippoc | `references_v4/HIPPOC_GSE295314` | 07-13, pre-relabel | no | 0.991 / 0.999 / 1.000 / 1.000 | 0.002 |
| kidney | `references_v4/KIDNEY_GSE240658` | 07-12, pre-relabel | no | 0.986 / 0.998 / 0.999 / 1.000 | 0.003 |
| lung_at2 | `references_v4/LUNG_native_pooled` | 07-06, pre-relabel | no | 0.978 / 0.996 / 0.999 / 0.999 | 0.003 |
| lung_club | `references_v4/LUNG_native_pooled` | 07-06, pre-relabel | no | 0.984 / 0.997 / 0.999 / 0.999 | 0.004 |
| skmgn | `references_v4/MUSCLE_GSE137869_Y` | 07-06, pre-relabel | no | 0.949 / 0.993 / 0.999 / 0.999 | 0.009 |
| smlint | `references_v4/SMLINT_GSE272055` | 07-27, pre-relabel | no | 0.992 / 0.999 / 1.000 / 1.000 | 0.006 |

r is the median Pearson-VST of the focal type's expression (`purity_sweep_summary.tsv`, `pearson_vst_median`).

- **Comparable sweeps:** the three that use the same reference in both arms agree to 8 decimals. The remaining
  difference comes from the old files rounding the truth fractions to 7 digits; the current code writes full
  precision.
- **The other eight sweeps:** these are new results, not a reproduction. The reference changed (the refs4 joint
  relabelling, 2026-10-27, and the TESTES gene-ID lift, 2026-10-22), so the default sweeps measure a different
  reference.
- **Chu-2022 bar:** all 11 sweeps on the current references meet it. At 50% purity the minimum is 0.992
  (HYPOTH); at 85% it is 0.998.
- **Correction:** `OPEN_FRONTIER.md` said that BAT / HYPOTH / SMLINT / TESTES reach "VST ≥0.998 at 50% purity"
  and that HYPOTH is "1.000 throughout". HYPOTH is 0.992 at 50%, in this rebuild and in the July sweep
  (identical), so the line was wrong. It is corrected.

**Not swept** (as before): BLOOD, WATSC, SPLEEN and OVARY have no `purity_sweep:` entry; LIVER is swept by
`run_purity_sweep.sh`; SKMVL shares SKMGN's reference, which is swept once.

## Notebooks

The pre-registration left the notebooks out of scope. They were run anyway, as a check that they follow the profile:
notebooks 8–12 were re-executed under the rebuild profile (job 12134960, `--out-dir`, all OK). Their text output
differs from the default run only in folder paths, run dates, the extra `seed` row of `de_methods.tsv`, and the
checksum of `posctrl_summary.md`, which contains its own path.

## What this establishes

- **Covered:** from the production references, the prepared bulk and the v2 checkpoint, the committed commands
  regenerate every Stage 8–13 output of the default byte for byte. The assembled default tree, built from links
  into three earlier runs, gives the same result as a fresh run.
- **BayesPrism:** with 16 cores and its default seed, it is deterministic across runs, including for BAT, HEART
  and HYPOTH, whose July runs recorded no settings.
- **GeneCompass embedding:** deterministic on the A100s used.
- **Not covered** (scope as pre-registered): the reference builds and joint re-annotation (§0.4–0.5), and
  Stages 1–7 (see `docs/repro_stages1_7/REPRODUCTION.md`).
