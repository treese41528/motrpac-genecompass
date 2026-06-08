#!/usr/bin/env python3
"""Emit deconvolution config values as shell `CFG_KEY=value` lines for the R/bash
wrappers, which have no YAML parser. PIPELINE_ROOT must be set by the caller (the
wrapper derives it from its own location). Best-effort: prints nothing on any
failure so the wrappers fall back to their repo-relative defaults (e.g. when
python/pyyaml is unavailable on a compute node).

Usage (in a wrapper):
  export PIPELINE_ROOT="$PROJECT_ROOT"
  command -v python3 >/dev/null && eval "$(python3 "$PROJECT_ROOT/deconvolution/_config_sh.py" 2>/dev/null || true)"
"""
import os
import sys

root = os.environ.get("PIPELINE_ROOT", ".")
sys.path.insert(0, os.path.join(root, "lib"))
try:
    from gene_utils import load_config, resolve_path
    c = load_config()
    d = c["deconvolution"]
    print(f'CFG_RAT_EXCLUDE_GENES={resolve_path(c, d["rat_exclude_genes"])}')
    if d.get("rat_protein_coding_genes"):
        print(f'CFG_RAT_PROTEIN_CODING_GENES={resolve_path(c, d["rat_protein_coding_genes"])}')
    print(f'CFG_PROTEIN_CODING_ONLY={"1" if d.get("protein_coding_only", False) else "0"}')
    if d.get("rat_sex_chrom_genes"):
        print(f'CFG_RAT_SEX_CHROM_GENES={resolve_path(c, d["rat_sex_chrom_genes"])}')
    print(f'CFG_EXCLUDE_SEX_CHROMOSOMES={"1" if d.get("exclude_sex_chromosomes", False) else "0"}')
    print(f'CFG_N_CORES={d.get("n_cores", 4)}')
    print(f'CFG_R_MODULE={d.get("r_module", "r/4.4.1")}')
    # MoTrPAC bulk gene-ID liftover (prepare_motrpac_bulk.sh) -- guarded so a missing
    # key skips only its own line, never the emissions above.
    if d.get("motrpac_bulk_dir"):
        print(f'CFG_MOTRPAC_DATA_DIR={resolve_path(c, d["motrpac_bulk_dir"])}')
    if d.get("rat_token_mapping"):
        print(f'CFG_RAT_TOKEN_MAPPING={resolve_path(c, d["rat_token_mapping"])}')
    _b = c.get("biomart", {})
    if _b.get("rat_gene_info"):
        print(f'CFG_RAT_GENE_INFO={resolve_path(c, _b["rat_gene_info"])}')
except Exception:
    pass
