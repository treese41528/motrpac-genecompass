#!/usr/bin/env python
"""04_tokenize.py -- tokenize the held-out cells as the v2 corpus was tokenized, and draw the evaluation set.

Implements the measurement inputs of STEP4_PREREG.md V3 ("the held-out cells, capped at 2,000 per study, drawn
with a fixed seed"). The tokenizer is the corpus's own Stage 5 code, called unchanged:
  pipeline/05_tokenization/tokenize_corpus.py :: load_token_dict, load_median_dict, build_eligible_gene_arrays,
                                                 process_file (-> tokenize_cell_batch)
with the committed default config (v2): raw counts / hybrid median (data/training/gene_medians_v2/
hybrid_gene_medians.pickle), log2(1+x), rank, top 2,048, token ids from data/training/ortholog_mappings/
rat_tokens.pickle, species = 2 (rat). process_file() takes its output folder as an argument, so nothing is
written under data/training/.

Checks before any held-out cell is tokenized:
  * the number of eligible genes (token and median) equals the v2 corpus's (stage5_manifest.json);
  * positive control: one training QC file (the smallest with at least 1,000 cells that the plan keeps with
    action 'keep') is tokenized again
    by this script and compared, column by column, with its rows in the v2 corpus shards. Any difference stops
    the script.

Then:
  tokenized/shards/task_<i>_shard_*.arrow    every cell of every kept held-out file (file_plan.tsv, action keep);
                                             i = the file's place in sorted sample_id order; files are tokenized
                                             in parallel (one file per process), which does not change a token
  tokenized/tokenize_manifest.json           per-file results, input md5s, the control's result
  tokenized/eval_set/                        HuggingFace dataset: at most 2,000 cells per study, drawn without
                                             replacement by numpy.random.default_rng([20261114, crc32(study)])
                                             over the study's cells in (sample_id, cell_index) order; a study
                                             with 2,000 cells or fewer contributes all of them
  record/step4_heldout/eval_draw.tsv         cells per study, cells drawn, seed

Usage: 04_tokenize.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as pa_ipc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C   # noqa: E402

os.chdir(C.ROOT)
sys.path.insert(0, str(C.ROOT / "pipeline/05_tokenization"))
sys.path.insert(0, str(C.ROOT / "lib"))
import tokenize_corpus as T   # noqa: E402
from gene_utils import load_config, resolve_path   # noqa: E402

COMPARE = ["input_ids", "values", "length", "species", "study_id", "cell_id", "sample_id", "gsm", "cell_index"]


def read_shards(shard_dir, names):
    """The rows of the named shards as one table; memory-mapped, so taking a few rows reads only those."""
    return pa.concat_tables([pa_ipc.open_stream(pa.memory_map(str(Path(shard_dir) / n), "r")).read_all()
                             for n in names])


def control(cfg, tok, top_n, cells_per_shard):
    """Re-tokenize one training file and compare with the v2 corpus shard rows of that file."""
    plan = [r for r in C.read_tsv(C.PLAN) if r["action"] == "keep"]
    pick = min((r for r in plan if int(r["n_cells"]) >= 1000), key=lambda r: (int(r["n_cells"]), r["sample_id"]))
    sid = pick["sample_id"]
    v2 = resolve_path(cfg, cfg["paths"]["tokenized_corpus_dir"])
    ref = None
    for mp in sorted((v2 / "manifests").glob("task_*_manifest.json")):
        for fr in C.load_json(mp)["file_results"]:
            if fr.get("sample_id") == sid:
                ref = read_shards(v2 / "shards", fr["shards_written"])
    if ref is None:
        raise RuntimeError(f"control: {sid} not found in the v2 task manifests")
    ref = ref.filter(pc.equal(ref["sample_id"], sid))
    out = C.out_path(C.TOK_DIR / "control_shards"); out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.arrow"):
        f.unlink()
    qc = resolve_path(cfg, cfg["paths"]["qc_h5ad_dir"]) / f"{sid}.h5ad"
    res = T.process_file(qc, tok["gene_index"], tok["tokens"], tok["medians"], top_n, cells_per_shard, out, 0, [0],
                         gsm_map={sid: pick["gsm"]})
    mine = read_shards(out, res["shards_written"])
    same = {c: mine[c].combine_chunks().equals(ref[c].combine_chunks()) for c in COMPARE}
    rec = {"sample_id": sid, "cells": mine.num_rows, "cells_v2": ref.num_rows, "columns_equal": same,
           "pass": mine.num_rows == ref.num_rows and all(same.values())}
    print("control:", json.dumps(rec), flush=True)
    if not rec["pass"]:
        raise SystemExit("control FAILED: this script does not reproduce the v2 tokenization; stopping")
    return rec


def main():
    cfg = load_config()
    if cfg.get("_profile"):
        sys.exit("PIPELINE_PROFILE is set; the held-out cells are tokenized with the committed default (v2) config")
    paths, ref_cfg = cfg["paths"], cfg.get("reference_assembly", {})
    top_n = int(ref_cfg.get("top_n_genes", T.TOP_N_DEFAULT))
    cells_per_shard = int(ref_cfg.get("cells_per_shard", 50_000))
    orth_dir, median_dir = resolve_path(cfg, paths["ortholog_dir"]), resolve_path(cfg, paths["median_dir"])
    v2 = resolve_path(cfg, paths["tokenized_corpus_dir"])
    v2_manifest = C.load_json(v2 / "stage5_manifest.json")
    if Path(v2_manifest["config_snapshot"]["median_dir"]).name != median_dir.name:
        sys.exit(f"median_dir {median_dir} is not the v2 corpus's ({v2_manifest['config_snapshot']['median_dir']})")

    ids, tokens, medians = T.build_eligible_gene_arrays(T.load_token_dict(orth_dir), T.load_median_dict(median_dir))
    tok = {"gene_index": T.build_gene_index(ids), "tokens": tokens, "medians": medians}
    n_v2 = v2_manifest["config_snapshot"]["n_eligible_genes"]
    if len(ids) != n_v2 or top_n != v2_manifest["config_snapshot"]["top_n_genes"]:
        sys.exit(f"eligible genes {len(ids)} / top_n {top_n} differ from the v2 corpus ({n_v2}, "
                 f"{v2_manifest['config_snapshot']['top_n_genes']})")
    ctrl = control(cfg, tok, top_n, cells_per_shard)

    plan = [r for r in C.read_tsv(C.REC / "file_plan.tsv") if r["action"] == "keep"]
    files = sorted(C.QC_DIR / f"{r['sample_id']}.h5ad" for r in plan)
    gsm_map = {r["sample_id"]: r["gsm"] for r in plan}
    shard_dir = C.out_path(C.TOK_DIR / "shards"); shard_dir.mkdir(parents=True, exist_ok=True)
    for f in shard_dir.glob("*.arrow"):
        f.unlink()
    t0 = time.time()

    def one(i, p):
        # one file per call; the file's index stands where the corpus run has its task number, so shard names
        # (task_<file index>_shard_<k>.arrow) cannot collide when files are tokenized in parallel
        return T.process_file(p, tok["gene_index"], tok["tokens"], tok["medians"], top_n, cells_per_shard, shard_dir,
                              i, [0], gsm_map=gsm_map)
    from joblib import Parallel, delayed
    # a worker holds up to 50,000 cells as Python lists before a shard is written (about 7 GB), so at most 6
    jobs = int(os.environ.get("HELDOUT_TOKENIZE_JOBS", min(6, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))))
    results = Parallel(n_jobs=jobs, backend="loky")(delayed(one)(i, p) for i, p in enumerate(files))
    for p, r in zip(files, results):                    # results come back in file order
        if r.get("status") != "ok":
            raise SystemExit(f"tokenization failed for {p.name}: {r}")
        print(f"{p.name}: {r['n_cells_written']:,} cells, {r['n_cells_empty']} empty", flush=True)

    # ---- the draw -----------------------------------------------------------------------------------------
    by_study = {}
    for r in results:                                   # results are in sorted sample_id order
        by_study.setdefault(r["study_id"], []).append(r)
    draw_rows, picked = [], []
    for study in sorted(by_study):
        frs = by_study[study]
        n = sum(r["n_cells_written"] for r in frs)
        idx = C.draw_cells(n, C.CAP_PER_STUDY, C.DRAW_SEED, study)
        offset = 0
        for r in frs:
            k = r["n_cells_written"]
            local = idx[(idx >= offset) & (idx < offset + k)] - offset
            if len(local):
                picked.append(read_shards(shard_dir, r["shards_written"]).take(pa.array(local)))
            offset += k
        draw_rows.append({"study": study, "files": len(frs), "cells_tokenized": n, "cells_drawn": len(idx),
                          "cells_not_drawn": n - len(idx), "cap": C.CAP_PER_STUDY, "seed": C.DRAW_SEED,
                          "rng": "numpy.random.default_rng([seed, crc32(study)]).choice(n, cap, replace=False)"
                                 if n > C.CAP_PER_STUDY else "all cells"})
    table = pa.concat_tables(picked).combine_chunks()
    import datasets
    eval_dir = C.out_path(C.TOK_DIR / "eval_set")
    if eval_dir.exists():
        import shutil
        shutil.rmtree(eval_dir)
    ds = datasets.Dataset(datasets.table.InMemoryTable(table))
    ds.save_to_disk(str(eval_dir))
    md5 = {f.name: C.file_md5(f) for f in sorted(eval_dir.iterdir()) if f.is_file()}
    C.write_tsv(C.REC / "eval_draw.tsv", draw_rows)

    manifest = {
        "tokenizer": "pipeline/05_tokenization/tokenize_corpus.py::process_file",
        "top_n_genes": top_n, "n_eligible_genes": len(ids), "species": 2,
        "inputs": {"rat_tokens.pickle": C.file_md5(orth_dir / "rat_tokens.pickle"),
                   "hybrid_gene_medians.pickle": C.file_md5(median_dir / "hybrid_gene_medians.pickle"),
                   "median_dir": str(median_dir), "ortholog_dir": str(orth_dir)},
        "v2_corpus_manifest": str(v2 / "stage5_manifest.json"),
        "control": ctrl,
        "n_files": len(results), "n_cells_tokenized": sum(r["n_cells_written"] for r in results),
        "n_cells_empty": sum(r["n_cells_empty"] for r in results),
        "eval_set": {"path": str(eval_dir), "cells": table.num_rows, "studies": len(by_study),
                     "cap_per_study": C.CAP_PER_STUDY, "seed": C.DRAW_SEED, "md5": md5},
        "elapsed_s": round(time.time() - t0, 1), "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
        "file_results": results,
    }
    with open(C.out_path(C.TOK_DIR / "tokenize_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"tokenized {manifest['n_cells_tokenized']:,} cells of {len(results)} files in {len(by_study)} studies; "
          f"evaluation set {table.num_rows:,} cells -> {eval_dir}")


if __name__ == "__main__":
    main()
