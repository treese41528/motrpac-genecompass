#!/usr/bin/env python3
"""Emit deconvolution config values as shell `CFG_KEY=value` lines for the R/bash
wrappers, which have no YAML parser. PIPELINE_ROOT must be set by the caller (the
wrapper derives it from its own location); PIPELINE_PROFILE is honoured by load_config.

A failure is FATAL: the error goes to stderr and the exit status is 3, so a wrapper
can never silently fall back to a hardcoded (arm-specific) path. Wrappers source
deconvolution/load_config.sh, which runs this with $DECONV_PYTHON (default python3)
and stops on a non-zero exit. Optional keys that are absent are simply not emitted.

Usage (in a wrapper):
  export PROJECT_ROOT PIPELINE_ROOT="$PROJECT_ROOT"
  source "$PROJECT_ROOT/deconvolution/load_config.sh"     # exports every CFG_* below
"""
import os
import shlex
import sys

root = os.environ.get("PIPELINE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root, "lib"))


def main():
    from gene_utils import load_config, resolve_path
    c = load_config()
    d = c["deconvolution"]
    out = []

    def emit(name, value):
        out.append(f"{name}={shlex.quote(str(value))}")

    emit("CFG_RAT_EXCLUDE_GENES", resolve_path(c, d["rat_exclude_genes"]))
    if d.get("rat_protein_coding_genes"):
        emit("CFG_RAT_PROTEIN_CODING_GENES", resolve_path(c, d["rat_protein_coding_genes"]))
    emit("CFG_PROTEIN_CODING_ONLY", "1" if d.get("protein_coding_only", False) else "0")
    if d.get("rat_sex_chrom_genes"):
        emit("CFG_RAT_SEX_CHROM_GENES", resolve_path(c, d["rat_sex_chrom_genes"]))
    emit("CFG_EXCLUDE_SEX_CHROMOSOMES", "1" if d.get("exclude_sex_chromosomes", False) else "0")
    emit("CFG_N_CORES", d.get("n_cores", 4))
    emit("CFG_R_MODULE", d.get("r_module", "r/4.4.1"))
    # Seed for the R steps of Stages 10-14 that draw random numbers (IHW folds, Augur, FDR sensitivities).
    emit("CFG_DE_SEED", int(d.get("de_seed", 1)))
    if d.get("motrpac_bulk_dir"):
        emit("CFG_MOTRPAC_DATA_DIR", resolve_path(c, d["motrpac_bulk_dir"]))
    if d.get("rat_token_mapping"):
        emit("CFG_RAT_TOKEN_MAPPING", resolve_path(c, d["rat_token_mapping"]))
    _b = c.get("biomart", {})
    if _b.get("rat_gene_info"):
        emit("CFG_RAT_GENE_INFO", resolve_path(c, _b["rat_gene_info"]))
    _r = c.get("rgd", {})
    if _r.get("genes_file"):
        emit("CFG_RAT_RGD_GENES", resolve_path(c, _r["genes_file"]))
    # Output dirs + committed artifacts (so R/bash source these from config, never a hardcoded arm).
    for key, env in [("motrpac_bulk_liftover", "CFG_MOTRPAC_BULK_LIFTOVER"),
                     ("motrpac_bulk_liftover_report", "CFG_MOTRPAC_BULK_LIFTOVER_REPORT"),
                     ("rat_genecompass_genes", "CFG_RAT_GENECOMPASS_GENES"),
                     ("built_reference_dir", "CFG_BUILT_REFERENCE_DIR"),
                     ("reference_v2_dir", "CFG_REFERENCE_V2_DIR"),
                     ("results_dir", "CFG_RESULTS_DIR"),
                     ("validation_dir", "CFG_VALIDATION_DIR"),
                     ("pilot_validation_dir", "CFG_PILOT_VALIDATION_DIR"),
                     ("validation_v2_dir", "CFG_VALIDATION_V2_DIR"),
                     ("motrpac_bulk_out", "CFG_MOTRPAC_BULK_OUT"),
                     ("genecompass_input_dir", "CFG_GENECOMPASS_INPUT_DIR"),
                     ("hotspots_table", "CFG_HOTSPOTS_TABLE"),
                     ("idspace_audit_dir", "CFG_IDSPACE_AUDIT_DIR"),
                     ("sample_pheno", "CFG_SAMPLE_PHENO"),
                     ("augur_input_dir", "CFG_AUGUR_INPUT_DIR"),
                     ("grn_dir", "CFG_GRN_DIR"),
                     ("grn_human_dir", "CFG_GRN_HUMAN_DIR"),
                     ("posctrl_prereg", "CFG_POSCTRL_PREREG")]:
        if d.get(key):
            emit(env, resolve_path(c, d[key]))
    print("\n".join(out))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # fatal: never let a wrapper fall back to a default arm
        print(f"FATAL: deconvolution/_config_sh.py could not read the pipeline config "
              f"(PIPELINE_ROOT={root}, PIPELINE_PROFILE={os.environ.get('PIPELINE_PROFILE', '')}): "
              f"{type(e).__name__}: {e}\n"
              f"  -> run with the project venv's python (export DECONV_PYTHON=<venv>/bin/python) "
              f"and check config/pipeline_config.yaml", file=sys.stderr)
        sys.exit(3)
