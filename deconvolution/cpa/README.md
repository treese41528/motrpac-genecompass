# CPA (Aim 2c / Module C) — isolated environment

**Why this directory exists.** Aim 2c builds a dose-response model with **`cpa-tools`**
(the maintained CPA — <https://github.com/theislab/cpa>), which pulls **scvi-tools +
pytorch-lightning + torch**. Installing that into the shared **`motrpac-env`** (Python 3.12;
TensorFlow 2.21 / GeneCompass / the entire deconvolution + embedding pipeline) risks breaking
the environment the whole project depends on. So **CPA lives in its own venv, `cpa-env`**, a
sibling of `motrpac-env`, fully isolated.

## Setup — once, safe by construction

```bash
bash deconvolution/cpa/setup_cpa_env.sh
```

This creates `/depot/reese18/apps/cpa-env` (a fresh venv, **no** `--system-site-packages`),
installs `cpa-tools`, verifies `cpa.CPA` + `cpa.CPA.setup_anndata` import, writes
`requirements-cpa.lock.txt`, and prints a check that `motrpac-env` was left **UNCHANGED**.
The script **refuses** to run if its target is (or resolves to) `motrpac-env`.

> Heavy download (torch + scvi-tools, ~9.4 GB). Run on a compute node if the login node is constrained:
> `srun --mem=16G --time=30 bash deconvolution/cpa/setup_cpa_env.sh` (no GPU needed to install).

**Status: built & verified** (2026-07-27) — `cpa-tools 0.8.8`, `scvi-tools 0.20.3`, `torch 2.0.0+cu117`,
Python 3.9.25; `import cpa` + `cpa.CPA.setup_anndata` confirmed. Three gotchas are handled by the
script/requirements — all reasons this must NOT go in `motrpac-env`:
1. **Stale PyPI** — install `cpa-tools` from **git** (the PyPI release pins ancient scvi-tools/PyYAML).
2. **Python 3.9–3.10** required (the shared env is 3.12) — the script uses `/usr/bin/python3.9`.
3. **`pyarrow<16`** — `scvi-tools → ray 2.9.3` uses `pyarrow.PyExtensionType`, removed in pyarrow ≥16.

## Use

```bash
source /depot/reese18/apps/cpa-env/bin/activate      # ← CPA work only
python deconvolution/run_cpa_dose.py ...             # (the driver, to be built)
deactivate
source /depot/reese18/apps/motrpac-env/bin/activate  # ← back to the pipeline env
```

## The golden rule for collaborators

**Never `pip install cpa-tools` (or scvi-tools / pytorch-lightning) into `motrpac-env`.**
Use `cpa-env`. There is no shared Python state to worry about: the CPA driver reads the
pipeline's outputs as plain files on disk —
`<genecompass_input_dir>/<tissue>/pseudocells.h5ad` (v2: `data/deconvolution/genecompass_input_v2/`),
`.../pseudobulk_de/de_hotspots.tsv`, `.../<TISSUE>/de__<celltype>.tsv` — and writes its own
outputs under `.../<tissue>/cpa/`. The two environments never need to be active at the same time.

## Can this be merged into `motrpac-env`? No — checked 2026-07-27

Two hard blockers plus a cascade of version conflicts make a single shared env impossible.
`cpa-tools 0.8.8` is anchored to the 2023-era **scvi-tools 0.20.3**, which drags an old ML stack:

| | motrpac-env | cpa-env |
|---|---|---|
| **Python** | **3.12.11** | **3.9.25** (cpa-tools `Requires-Python: >=3.9,<3.11`) |
| **numpy** | **2.4.3** (TensorFlow 2.21) | **1.23.5** (numpy 1.x ↔ 2.x is an ABI break) |
| torch | 2.11.0 | 2.0.0 |
| anndata | 0.12.10 | 0.9.2 |
| pyarrow | 23.0.1 | 15.0.2 |
| scikit-learn / numba / scipy | 1.8.0 / 0.64.0 / 1.17.1 | 1.6.1 / 0.58.1 / 1.12.0 |

Merging would require downgrading numpy (breaks TensorFlow), downgrading torch (breaks GeneCompass),
and moving to Python 3.10 (rebuilds the whole pipeline) — i.e. destroying `motrpac-env`. The only future
path to one env is if `cpa-tools` upstream migrates to **scvi-tools ≥1.x** (py3.10–3.12, numpy 2). Until
then, keep them separate.

## Files here

| File | Role |
|---|---|
| `setup_cpa_env.sh` | Builds the isolated `cpa-env` (idempotent; guards against touching `motrpac-env`). |
| `requirements-cpa.txt` | Top-level requirement (`cpa-tools`). |
| `requirements-cpa.lock.txt` | Exact frozen pins, written by the setup script after a successful build. |
| `README.md` | This file. |

See `manuscript/wip/wip_aim2c_cpa_dose_response.md` for the full Module-C plan and the modern
`cpa.CPA` API mapping (drug→exercise, dose→week, sex as a subtractable covariate).
