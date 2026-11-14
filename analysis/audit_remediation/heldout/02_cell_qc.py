#!/usr/bin/env python
"""02_cell_qc.py -- Stage 2 cell QC of the held-out series, into a folder of their own.

Implements STEP5_PREREG.md M2 ("put through the Stage 2 cell QC, into a folder of their own (never the
training tree)"). The QC is the corpus's own code, called unchanged:
  pipeline/02_gene_universe/preprocess_training_matrices.py :: process_study()
with the committed config (config/pipeline_config.yaml: gene_universe.preprocessing thresholds), the corpus's
gene universe and resolution tables (data/training/gene_universe/gene_universe.tsv, gene_resolution.tsv; read
only) and the same file discovery and numbering (find_matrix_files, number_matrix_files). The only difference
from a corpus run is the output folder, which process_study() takes as an argument:
  data/audit_remediation/heldout/qc_h5ad/<series>_sample<i>.h5ad      QC'd cells, raw counts in X
  data/audit_remediation/heldout/qc_reports/report_<series>.json      process_study()'s report
  data/audit_remediation/heldout/qc_reports/provenance_<series>.json  thresholds, input md5s, commit
Nothing is normalised, nothing is written under data/training/. No expression_stats.tsv is written (the
held-out cells must not feed the gene universe).

The series come from record/step4_heldout/series_list.tsv (column heldout_series, written by
01_series_list.py). A series whose report exists is skipped, so the script can be rerun.

Usage: 02_cell_qc.py --group I --n-groups N [--jobs J]     series split into N size-balanced groups
       02_cell_qc.py --studies GSE1 GSE2 ... [--jobs J]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C   # noqa: E402

sys.path.insert(0, str(C.ROOT / "pipeline/02_gene_universe"))
sys.path.insert(0, str(C.ROOT / "lib"))


def heldout_series():
    return sorted(r["accession"] for r in C.read_tsv(C.REC / "series_list.tsv") if r["heldout_series"] == "True")


def raw_size(acc, P, cfg):
    files = P.number_matrix_files(acc, P.find_matrix_files(str(C.GEO / acc)), cfg)
    return sum(os.path.getsize(fp) for _, fp, _ in files)


def groups(series, n, P, cfg):
    """Greedy split by raw matrix size (largest first into the lightest group); deterministic."""
    sized = sorted(((raw_size(a, P, cfg), a) for a in series), reverse=True)
    out, load = [[] for _ in range(n)], [0] * n
    for sz, a in sized:
        i = load.index(min(load))
        out[i].append(a); load[i] += sz
    return out


def run_one(acc):
    """QC one series in its own process."""
    os.environ.setdefault("PIPELINE_ROOT", str(C.ROOT))
    import preprocess_training_matrices as P
    from gene_utils import load_config, resolve_path
    cfg = load_config()
    if cfg.get("_profile"):
        raise RuntimeError("PIPELINE_PROFILE is set; the held-out QC runs on the committed default config")
    qc_dir = C.out_path(C.QC_DIR); rep_dir = C.out_path(C.QC_REPORTS)
    qc_dir.mkdir(parents=True, exist_ok=True); rep_dir.mkdir(parents=True, exist_ok=True)
    rep_path = rep_dir / f"report_{acc}.json"
    if rep_path.is_file():
        return acc, "skipped (report exists)", 0
    gu_dir = resolve_path(cfg, cfg["paths"]["gene_universe_dir"])
    mapper = P.GeneUniverseMapper(gu_dir)
    t0 = time.time()
    report, _gene_stats = P.process_study(study_accession=acc, study_dir=str(C.GEO / acc), mapper=mapper,
                                          config=cfg, output_dir=str(qc_dir))
    prov = {"series": acc, "qc_code": "pipeline/02_gene_universe/preprocess_training_matrices.py::process_study",
            "preprocessing_config": cfg["gene_universe"].get("preprocessing", {}),
            "gene_universe_dir": str(gu_dir), **P._step2_provenance(gu_dir),
            "output_dir": str(qc_dir), "elapsed_seconds": round(time.time() - t0, 1),
            "slurm_job": os.environ.get("SLURM_JOB_ID", "")}
    with open(rep_dir / f"provenance_{acc}.json", "w") as f:
        json.dump(prov, f, indent=1, default=str)
    tmp = rep_path.with_suffix(".json.part")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=1, default=str)
    os.replace(tmp, rep_path)                     # the report is the 'done' marker: written last, atomically
    return acc, report["status"], report.get("total_cells_after_qc", 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--studies", nargs="*")
    ap.add_argument("--group", type=int)
    ap.add_argument("--n-groups", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--list", action="store_true", help="print the groups and exit")
    a = ap.parse_args()

    import preprocess_training_matrices as P
    from gene_utils import load_config
    cfg = load_config()
    series = heldout_series()
    if a.studies:
        bad = sorted(set(a.studies) - set(series))
        if bad:
            sys.exit(f"not held-out series: {bad}")
        todo = a.studies
    else:
        g = groups(series, a.n_groups, P, cfg)
        if a.list:
            for i, x in enumerate(g):
                print(i, len(x), f"{sum(raw_size(s, P, cfg) for s in x) / 1e9:.2f} GB", " ".join(x))
            return
        todo = g[a.group]
    print(f"QC of {len(todo)} series with {a.jobs} worker(s): {' '.join(todo)}", flush=True)
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=a.jobs, backend="loky")(delayed(run_one)(s) for s in todo)
    for acc, status, cells in res:
        print(f"{acc}\t{status}\t{cells}")


if __name__ == "__main__":
    main()
