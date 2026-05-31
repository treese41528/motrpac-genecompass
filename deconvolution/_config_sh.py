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
    print(f'CFG_N_CORES={d.get("n_cores", 4)}')
    print(f'CFG_R_MODULE={d.get("r_module", "r/4.4.1")}')
except Exception:
    pass
