"""Paths and per-tissue facts for the Stage 8-12 notebooks, taken from the active config.

PIPELINE_PROFILE is honoured, so running a profile with notebooks/run_notebooks.py shows that profile's outputs.
Every per-tissue number the notebooks show comes from the outputs themselves: provenance, summaries, the
tokenized datasets. The tissue list is the set of deconvolved tissues in results_dir.

    from nb_config import RES_DIR, GC_DIR, GC_HUMAN_DIR, DE_DIR, GATE_DIR, GATE_HUMAN_DIR, TISSUES, tissues
"""
import csv
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "lib"))
from gene_utils import load_config, resolve_path  # noqa: E402

os.environ.setdefault("PIPELINE_ROOT", str(PROJECT))   # the kernel runs in notebooks/, not the repo root
CFG = load_config(str(PROJECT / "config/pipeline_config.yaml"))
_D = CFG["deconvolution"]
RES_DIR = resolve_path(CFG, _D["results_dir"]) / "motrpac"                 # Stage 8, one folder per TISSUE
GC_DIR = resolve_path(CFG, _D["genecompass_input_dir"])                    # Stage 9, one folder per tissue
GC_HUMAN_DIR = Path(str(GC_DIR) + "_human")                                # Stage 12 (as run_stage12.py names it)
DE_DIR = GC_DIR / "pseudobulk_de"                                          # Stage 10
GATE_DIR = resolve_path(CFG, _D["hotspots_table"]).parent                  # Stage 11 gate (gate_rcv10/)
GATE_HUMAN_DIR = GC_HUMAN_DIR / GATE_DIR.name
PROFILE = CFG.get("_profile") or "(default config)"

TISSUES = sorted(p.name for p in RES_DIR.iterdir() if (p / "estimated_fractions.csv").is_file())
tissues = [t.lower() for t in TISSUES]

# run.prism minutes for runs made before run_deconvolution.R recorded them in provenance (2026-11-01). Each row
# was read from that run's SLURM log; it applies only to the results_dir the row names.
_PRISM_BACKFILL = PROJECT / "deconvolution/reference/run_prism_minutes_backfill.tsv"


def prism_minutes(tissue: str):
    """(minutes, source) for one tissue's BayesPrism run.prism, or (nan, 'not recorded')."""
    prov = RES_DIR / tissue / "provenance_run_deconvolution.json"
    if prov.is_file():
        m = json.loads(prov.read_text()).get("params", {}).get("run_prism_minutes")
        if m is not None:
            return float(m), "provenance"
    if _PRISM_BACKFILL.is_file():
        here = str(RES_DIR.parent.relative_to(PROJECT))
        for r in csv.DictReader((l for l in open(_PRISM_BACKFILL) if not l.startswith("#")), delimiter="\t"):
            if r["results_dir"] == here and r["tissue"] == tissue:
                return float(r["minutes"]), f"log {r['log']}"
    return float("nan"), "not recorded"


def tokenize_summary(tissue: str, human: bool = False) -> dict:
    f = (GC_HUMAN_DIR if human else GC_DIR) / tissue.lower() / "tokenize_summary.json"
    return json.loads(f.read_text()) if f.is_file() else {}


def transfer_summary(tissue: str) -> dict:
    f = GC_HUMAN_DIR / tissue.lower() / "transfer_summary.json"
    return json.loads(f.read_text()) if f.is_file() else {}


def n_embedded(tissue: str, human: bool = False):
    """Rows of the tissue's cell_embeddings.npy (the pseudo-cells actually embedded)."""
    import numpy as np
    f = (GC_HUMAN_DIR if human else GC_DIR) / tissue.lower() / "embeddings" / "cell_embeddings.npy"
    return int(np.load(f, mmap_mode="r").shape[0]) if f.is_file() else None


def min_token_length(tissue: str, human: bool = False):
    """Shortest tokenized pseudo-cell (cells under 200 genes are dropped before this, see n_dropped_lt_min_genes)."""
    from datasets import load_from_disk
    d = (GC_HUMAN_DIR if human else GC_DIR) / tissue.lower() / "dataset"
    if not d.is_dir():
        return None
    lengths = load_from_disk(str(d))["length"]           # stored as a one-element list per cell
    return int(min(x[0] if isinstance(x, list) else x for x in lengths))


print(f"config: {PROFILE}\nStage 8: {RES_DIR}\nStage 9: {GC_DIR}\n{len(TISSUES)} tissues: {', '.join(TISSUES)}")
