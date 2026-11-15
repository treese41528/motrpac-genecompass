# Deconvolution (grant Aim 2 bridge)

**The question.** MoTrPAC's rat RNA-seq is bulk: one value per gene for a whole piece of tissue. Can it be
split, tissue by tissue, into per-cell-type expression that is accurate enough to ask which cell types and
which genes respond to training?

**The chain.** MoTrPAC bulk RNA-seq → **BayesPrism** deconvolution against a rat single-cell reference →
one **pseudo-cell** per sample and cell type → fine-tuned **rat GeneCompass** → 768-d cell embeddings → the
gate, the differential expression, the transfer and the networks.

**The result, on the default configuration (refs5, since 2026-11-15).** refs5 is the registered rerun of
Stages 8–14 that followed the 2026-11-11 audit, with the v2 model. Sources:
`docs/audit/remediation/STEP3_RESULTS.md` (the rerun) and `STEP4_RESULTS.md` (the validations); every number
there has a table under `docs/audit/remediation/record/`.

- **What ran.** 16 tissues, 749 samples, 15 references, 194 tissue × cell-type blocks, of which 150 are
  analysable (median per-sample library of at least 1,000); 9,060 pseudo-cells embedded. A repeat of Stage 8
  from a second commit is byte-identical.
- **What the deconvolution supports.** How a cell type's fraction varies across samples: on known-truth
  mixtures BayesPrism meets the registered floor on 19 of 19 sets (median per-type r 0.976 to 0.999), with 6
  of 192 type tests below 0.85 (LIVER hepatocytes, OVARY erythroid cells, HIPPOC cross-study excitatory
  neurons, three KIDNEY cross-study types). The expression of each tissue's dominant type: 12 of 15 pass
  their purity sweep.
- **What it does not support.**
  - *Fractions as cell proportions.* BayesPrism estimates the share of mRNA. Its error against cell fractions
    rises under the references' own mRNA differences on 13 of 15 references.
  - *The expression of rare types.* Expression is recovered at the type's own abundance for 4 of the 12 called
    blocks and for none of the five muscle blocks (correlation 0.50 to 0.57 at 0.5% of a mixture). BLOOD
    erythroid cells and SPLEEN red-pulp macrophages fail the dominant-type sweep.
  - *Specific tissues.* HIPPOC's composition stays flagged unreliable (decision of 2026-11-14). TESTES
    levels depend on the reference (a control-only reference keeps the order of animals and shifts levels by
    up to 5.6 points). One HYPOTH sample stands apart from the other 49. 50.6% of the SMLINT reference's
    cells and 16.7% of the KIDNEY reference's are above 15% mitochondrial reads (`STEP5_RESULTS.md`, M1).
- **What was found downstream on these pseudo-cells.** 11 exercise hotspots and 1 borderline block, with a
  list that is not stable to small changes of the inputs; 52,961 dose-significant gene × block pairs (9,617
  distinct genes), none of whose blocks is robust to technical covariates by the registered verdict; the
  positive-control red flag stands (5 of 26 testable pipeline-sanity controls); 10 of 11 hotspots transfer
  by the registered rule and 9 by the corrected rule; the 7 perturbation networks carry no TF-specific signal.
  No called block passes all nine registered readings.

**Where things are.** Read the folders from `config/pipeline_config.yaml` (section `deconvolution`), not from
this page:

| Config key | Holds |
|---|---|
| `reference_dir` of each tissue in `tissue_references.yaml` | The reference (refs5: `data/deconvolution/references_v5/`; BLOOD is a whole-blood composite) |
| `rat_exclude_genes` | The genes left out of the deconvolution (refs5: `reference/rat_exclude_genes_v2.tsv`) |
| `results_dir` | Stage 8: fractions, Z and the sample key per tissue |
| `genecompass_input_dir`, and the same with `_human` | Stages 9, 10, 12: pseudo-cells, embeddings, `pseudobulk_de/`; the human-space copies |
| `hotspots_table` | Stage 11: the gate table; its folder (`gate_rcv10/`) holds every gate output. A hotspot is `gate_class == hotspot` |
| `pseudocell_target_sum` | Stages 9 and 12: 6,700 for refs5, by the registered formula |
| `augur_input_dir`, `grn_dir`, `grn_human_dir`, `validation_dir` | Augur exports; Stage 13 networks; Stage 14 sweeps |

Earlier arms are profiles: `config/profiles/refs4all_rebuild.yaml` (the default before refs5; results in
`ALL_TISSUE_RERUN_RESULTS.md` and `REFS4ALL_REBUILD_RESULTS.md`), `refs4_14.yaml` (14 tissues;
`REFS4_RERUN_RESULTS.md`), `july_refs.yaml` (the July references), `april.yaml` (April's model). Their
results documents are dated records; each carries a banner saying which run it describes. Rebuild commands:
`DECONV_REBUILD_RUNBOOK.md` §0. Bulk gene-ID preparation: `MOTRPAC_BULK_LIFTOVER.md`.

**Reference integrity — start here.** `tissue_references.yaml` is the canonical tissue→reference map
(single source of truth). `REFERENCE_QC.md` documents the build-time QC gate (`reference_qc.py`) and the
two reference bugs it now prevents: the **liver** Visium-spatial contamination (removed) and the
**engineered lung** reference (GSE178405 → native pooled `lung_native_pooled`). The gate blocks spatial,
engineered or cultured, and bulk samples by their inventory titles; it only warns on developmental ages and
mixed depth, and it cannot test samples outside the inventory (BLOOD, SPLEEN, OVARY). All 15 references were
rebuilt from committed code for refs5 and 13 came out byte-identical; HYPOTH is re-annotated and HEART has its
registered label merge. The label decisions were made by an AI assistant and no specialist has read them.
`OMNIDECONV_RESULTS.md` holds the earlier cross-method runs and the guidance on which downstream claims to
trust; the runs on the current references are `docs/audit/remediation/STEP4_RESULTS.md` V5 (mRNA-content
bias) and V6 (mixture panel). The analysis notebooks (`../notebooks/pipeline8–12`) read the active config and
compute every number they show: after a rerun they are re-executed
(`python notebooks/run_notebooks.py 8 9 10 11 12`), not edited; see the banner of
`../notebooks/RERUN_EDITS.md`.

## Pipeline (production path)

The orchestrators in `pipeline/` drive the scripts in this folder in place
(nothing is moved), in the same `run_stageN.py` style as the rest of the pipeline:

| Stage | Orchestrator | Chain (scripts in this folder unless a path is given) |
|---|---|---|
| **8 — Deconvolution** | `pipeline/run_stage8.py` | `R/prepare_motrpac_bulk.sh` → `R/run_deconvolution.sh` → `R/extract_z.sh` → `build_pseudocells.py` |
| **9 — Tokenize + Embed** | `pipeline/run_stage9.py` | `tokenize_pseudocells.py` → `finetune/genecompass/embed_cells.py` |
| **11 — Gate** | `pipeline/run_stage11.py` | `embed_qc.py` → `pheno_merge_test.py` → `subspace_probe.py --sequential` → `augur_prep.py` → `run_augur.sh` → `corroborate_summary.py` |
| **10 — DE** | `pipeline/run_stage10.py` | `R/run_pseudobulk_de.sh` → `compare_posctrl.py` → with `--covariate-variants A B`, the whole DE again with technical covariates and `compare_de_runs.py` |
| **12, 13, 14** | `pipeline/run_stage12.py`, `run_stage13.py`, `run_stage14.py` | transfer (`translation/`); networks, their decoy null and enrichment; purity sweeps and the composition table |

Stage 8 = bulk → `pseudocells.h5ad`; Stage 9 = pseudo-cells → `cell_embeddings.npy`.
Both are per-tissue; each prints its plan with `--dry-run` and supports `--from N`. Stages 10–14 run once
over all tissues, in the order 11, 10, 12, 13, 14.

```bash
# one tissue, end to end (step 2 of stage 8 is CPU-heavy -> compute node;
# stage 9 embed -> GPU node):
python pipeline/run_stage8.py --tissue SKM-GN \
    --ref-dir "data/deconvolution/references_v5/MUSCLE_GSE137869_Y"
python pipeline/run_stage9.py --label skmgn
```

You supply `--tissue` (MoTrPAC bulk code, matches `TRNSCRPT_<TISSUE>_RAW_COUNTS.rda`)
and `--ref-dir` (the per-tissue SC reference). The tissue→reference mapping is
**`tissue_references.yaml`** — the canonical single source of truth (each tissue's correct reference +
study + build command). The bulk code differs from the reference dir name (e.g. `LUNG` →
`references_v5/LUNG_native_pooled`, `LIVER` → `references_v5/LIVER_GSE220075`, `BLOOD` →
`references_v5/BLOOD_composite`).

**Recommended — from-scratch, correct-by-construction:** `pipeline/run_deconv_all.py` reads that
manifest, runs the QC gate (`reference_qc.py --fail`) on every reference — refusing a missing reference or
one the gate blocks, so a fresh run cannot silently use the wrong study — and submits Stage-8+9 for all 16
tissues with the right `--ref-dir`:

```bash
python pipeline/run_deconv_all.py --dry-run     # validate every reference + print the plan
python pipeline/run_deconv_all.py --submit      # sbatch one job per tissue (refs from the manifest)
# then the cross-tissue layer (once all tissues finish): gate (subspace_probe + Augur +
# corroborate_summary -> hotspots), Stage 10 DE, Stage 12 transfer, Stage 13, Stage 14.
# Exact commands: DECONV_REBUILD_RUNBOOK.md §0. (Local SLURM wrappers: slurm/analysis/.)
```

References are **built, not shipped** — `build_references_from_config.py` builds every reference in
`tissue_references.yaml` into `references_v5/` (joint labels where the reference has them; `build_blood_composite.py` for BLOOD;
`build_lung_pooled.py` for the native lung pool; the liver QC gate auto-drops the Visium samples). So a fresh clone must set up data first — see [`setup/SETUP.md`](setup/SETUP.md) (Data & config).
Other paths default from `config/pipeline_config.yaml` `[deconvolution]`; outputs land under
`data/deconvolution/` (gitignored).

## Not in the numbered stages (validation / exploration)

These are run on demand, separately from the stage orchestrators:

- **Viewer:** `build_umap_viewer.py` (+ `umap_viewer_template.html`, `umap_embeddings.py`). No viewer
  exists for the default, and the viewer's "hotspot" rule is not the gate's.
- **Deconvolution validation:** `make_pseudobulk.py`, `make_purity_sweep.py`,
  `score_purity_sweep.py`, `score_validation.py`, `score_z.py`, `compute_true_z.py`,
  `R/score_z_vst.R`, `R/run_tutorial_gbm.R`, `build_all_datasets.sh`.
- **Cross-method θ check + mRNA-bias battery:** `R/run_omnideconv.R` +
  `omnideconv_bench/` (the SimBu simulator `simulate_simbu.{R,sh}`, `bias_delta.py`, `dose_response.py`).
  On the current references: `docs/audit/remediation/STEP4_RESULTS.md` V5 and V6 (full report
  `STEP4_V5_V6_RESULTS.md`). Earlier runs, the paper comparison and the downstream-claims guidance:
  `OMNIDECONV_RESULTS.md` (plan: `OMNIDECONV_BENCHMARK_PLAN.md`).
- **Reference build + QC:** `build_reference.py` (with the built-in QC gate), `build_references_v2.sh`,
  `build_lung_pooled.py` (native-lung pool); **`reference_qc.py`** (the contamination gate — run
  `python deconvolution/reference_qc.py --all --deep` to audit every reference); **`tissue_references.yaml`**
  (the map). Plus `build_protein_coding_list.py`, `build_sex_chrom_list.py`, `audit_idspace.py`,
  `setup/` (container + bootstrap). References are inputs to Stage 8.

## Setup & environment

**First-time setup (env, site profile, data, config): see
[`setup/SETUP.md`](setup/SETUP.md)** — the authoritative Stage 8–9 runbook.

- **Python** steps run under the `motrpac-env` venv (`source` it, or set `DECONV_PYTHON` /
  `MGC_PYTHON`).
- **R** steps run via their `R/*.sh` wrappers, which set `R_LIBS_USER=R_libs/` and source
  `setup/site_env.sh` to reproduce the build env from the per-site `setup/site.env` (load
  `R_MODULES`, strip `STRIP_CONDA`, resolve the python). All of it is **conditional**, so the
  wrappers are portable — a no-op off a module cluster / inside the container. Prefer the
  wrappers over the `.R` files directly.
- **Config** is shared across both languages: Python via `lib/gene_utils.load_config`,
  R/bash via `_config_sh.py` (emits `CFG_*` shell vars from the same YAML). Machine-specific
  path overrides go in the gitignored `config/pipeline_config.local.yaml` (template:
  `config/pipeline_config.local.yaml.example`); cluster/python paths go in `setup/site.env`.

## CIBERSORTx on an HPC cluster (license token + IP binding)

CIBERSORTx is the one license-gated method in the cross-method panel (`R/run_omnideconv.R`,
method `cibersortx`). It is **closed-source** and distributed as a Docker image
([`cibersortx/fractions`](https://hub.docker.com/r/cibersortx/fractions)) gated by registration +
a per-account **token**.

**Links & licensing**
- Register + generate the token: <https://cibersortx.stanford.edu> (log in → *Account* / "Download Token").
- Docker image: `docker pull cibersortx/fractions` ([Docker Hub](https://hub.docker.com/r/cibersortx/fractions)).
- **License:** CIBERSORTx is provided by the Alizadeh/Newman labs (Stanford) and is **free for
  academic / non-profit / government, non-commercial use only**; commercial use requires a separate
  license from Stanford. Review the Terms of Use on the site before use — we redistribute **no**
  CIBERSORTx code or the container image in this repo (only our `run_omnideconv.R` wrapper); each user
  must register and obtain their own token. The `token.txt` and `.sif` are gitignored.
- **Cite:** Newman A.M. *et al.*, "Determining cell type abundance and expression from bulk tissues
  with digital cytometry," *Nat. Biotechnol.* 37, 773–782 (2019); and the original CIBERSORT:
  Newman A.M. *et al.*, *Nat. Methods* 12, 453–457 (2015).

Two HPC-specific gotchas make it awkward — neither is cluster-vendor-specific:

**1. The token is bound to the IP address that contacts the license server *at run time*.**
The CIBERSORTx container phones home to validate `(email, token, IP)` every run. The bound IP is
the **egress (NAT) IP** the license server sees — which on a cluster is usually *not* the IP your
browser used to generate the token. So a token minted on your laptop/login node fails on a compute
node with `Token/username/ip combination is invalid … your IP address has changed`.
To fix it, mint the token from a context that egresses via the **same** IP your jobs will use:
  - Find the compute egress IP from inside a batch job: `curl https://api.ipify.org`. Check it's
    **stable across nodes** (run it on several) — clusters often NAT all compute nodes through one IP,
    but login nodes and compute nodes typically differ, and some clusters round-robin.
  - Generate the token through a browser **tunneled to egress via that IP** — e.g. an SSH dynamic
    SOCKS proxy (`ssh -D <port> …`) through a node whose egress matches where jobs run, with the
    browser pointed at `socks5://127.0.0.1:<port>` (enable remote DNS). Confirm with
    `https://api.ipify.org` in the proxied browser before generating.
  - The token is **reusable as long as that egress IP is stable** (here it's good for ~6 months),
    so this is a one-time setup. Browser notes: Chromium honors the Windows/system cert store (like
    `curl`) and may need `--disable-quic` over a TCP SOCKS proxy; Firefox uses its own cert store
    (`security.enterprise_roots.enabled=true` to trust system roots) and may need OCSP soft-fail.

**2. Run the container with a writable HOME + clean env (Docker is usually unavailable on HPC → use
Apptainer/Singularity).** `apptainer pull docker://cibersortx/fractions` → `fractions_latest.sif`.
The container ships **its own R** for the nu-SVR signature-matrix build, which needs a writable
`$HOME` and an uncontaminated environment; running it `--no-home`/`--contain` (omnideconv 0.1.1's
default) breaks that R (`there is no package called 'e1071'` → empty matrix → abort 134). Run it as:
```
apptainer exec --cleanenv --home <writable_dir> \
  -B <input_dir>:/src/data -B <output_dir>:/src/outdir \
  fractions_latest.sif /src/CIBERSORTxFractions --single_cell TRUE \
  --username <email> --token <token> --refsample <sig> --mixture <bulk> ...
```
`--cleanenv` also drops any host `LD_PRELOAD`/`R_*` vars that would otherwise leak into the
container. `R/run_omnideconv.R` does all of this for you: it reads `CIBERSORTX_EMAIL` and
`CIBERSORTX_TOKEN` from the environment (so the token never lands in code or, with `verbose=FALSE`,
in logs), pulls/uses the `.sif` under `CIBERSORTX_SIF_DIR`, and patches omnideconv's container
command to the `--cleanenv --home` form above. Invoke with
`OMNIDECONV_METHODS=…,cibersortx CIBERSORTX_EMAIL=… CIBERSORTX_TOKEN=$(cat token.txt) … run_omnideconv.sh`.
