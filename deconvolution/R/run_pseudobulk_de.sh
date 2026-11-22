#!/bin/bash
# Wrapper for run_pseudobulk_de.R -- differential expression per (tissue x cell type) on BayesPrism Z
# (limma-trend on log2-CPM; the models are listed in the R script's header and in de_methods.tsv).
# Same R env as the other R/*.sh wrappers (project R_libs/, suppressed site Rprofile, site profile for
# modules/conda). It reads pred_z already on disk for all tissues; no deconvolution is rerun.
# A full run (16 tissues) takes about 15 minutes and needs a compute node on this cluster (run it inside an
# sbatch job, or through pipeline/run_stage10.py inside one).
#
# Usage:  run_pseudobulk_de.sh [TISSUE ...]      (default: all tissues with pred_z; hotspots first)
#   e.g.  run_pseudobulk_de.sh                   # full run, every tissue
#         run_pseudobulk_de.sh BLOOD SKMVL       # just these tissues (debugging: IHW and repfdr are pooled)
# Override inputs and outputs via env: RESULTS_ROOT, BULK_ROOT, PHENO, HOTSPOTS, OUT_DIR.
# Settings via env (pipeline/run_stage10.py sets them from its options):
#   DE_SEED             IHW fold seed                         (--seed; default deconvolution.de_seed, else 1)
#   DE_ALPHA            significance level of the R step       (--alpha; default 0.05)
#   DE_TECH_COVARIATES  comma-separated technical covariates added to every model, among
#                       rin, pct_globin, pct_umi_dup, median_5_3_bias   (--tech-covariates; default none)
#   DE_MIN_MEDIAN_LIBSIZE  analysable-block rule              (--sensitivities; default 1000)
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT PIPELINE_ROOT="$PROJECT_ROOT"
# Config read is fatal on failure (no fallback to a hardcoded arm); exports every CFG_*.
source "${PROJECT_ROOT}/deconvolution/load_config.sh"
export R_LIBS_USER="${PROJECT_ROOT}/R_libs"
export TMPDIR="${PROJECT_ROOT}/tmp"
export R_PROFILE=/dev/null R_PROFILE_USER=/dev/null R_ENVIRON=/dev/null R_ENVIRON_USER=/dev/null
mkdir -p "${TMPDIR}"

# Reproduce the build/run env via the shared site profile: load modules (R_MODULES) and
# strip conda (STRIP_CONDA) only where site.env declares them and Lmod exists; a no-op on a
# laptop/container, where R comes from PATH. See deconvolution/setup/site_env.sh + SETUP.md.
source "${PROJECT_ROOT}/deconvolution/setup/site_env.sh"

# Paths come from the config (load_config.sh above is fatal on failure); each is overridable via env.
GC_DIR="${CFG_GENECOMPASS_INPUT_DIR:?genecompass_input_dir missing from the config}"
RESULTS_ROOT="${RESULTS_ROOT:-${CFG_RESULTS_DIR:?results_dir missing from the config}/motrpac}"
BULK_ROOT="${BULK_ROOT:-${CFG_MOTRPAC_BULK_OUT:?motrpac_bulk_out missing from the config}}"
PHENO="${PHENO:-${CFG_SAMPLE_PHENO:-${PROJECT_ROOT}/deconvolution/reference/motrpac_sample_pheno.tsv}}"
HOTSPOTS="${HOTSPOTS:-${CFG_HOTSPOTS_TABLE:-${GC_DIR}/gate_rcv10/corroboration_merged.tsv}}"
OUT_DIR="${OUT_DIR:-${GC_DIR}/pseudobulk_de}"
# IHW fold seed (config deconvolution.de_seed, default 1; env DE_SEED or run_stage10.py --seed override)
export DE_SEED="${DE_SEED:-${CFG_DE_SEED:-1}}"
# Significance level and technical covariates of the R step (STEP3_PREREG.md 6.1 and 6.4)
export DE_ALPHA="${DE_ALPHA:-0.05}"
export DE_TECH_COVARIATES="${DE_TECH_COVARIATES:-}"

echo "results : ${RESULTS_ROOT}"
echo "bulk    : ${BULK_ROOT}"
echo "out     : ${OUT_DIR}"
echo "seed    : ${DE_SEED}"
echo "alpha   : ${DE_ALPHA}"
echo "covars  : ${DE_TECH_COVARIATES:-none}"
Rscript --no-init-file --no-site-file \
  "${PROJECT_ROOT}/deconvolution/R/run_pseudobulk_de.R" \
  "${RESULTS_ROOT}" "${BULK_ROOT}" "${PHENO}" "${HOTSPOTS}" "${OUT_DIR}" "$@"
