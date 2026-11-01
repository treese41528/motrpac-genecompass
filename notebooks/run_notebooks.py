#!/usr/bin/env python3
"""run_notebooks.py -- re-execute the per-stage analysis notebooks (notebooks/pipelineN_analysis.ipynb).

Runs each notebook top to bottom with nbclient (the engine behind `jupyter nbconvert --execute`) in a kernel of
THIS interpreter, so launch it with the project venv's python: the venv's own `python3` kernelspec resolves to
the running interpreter, with no user-level kernel registration needed. The notebooks read stage outputs from
the active config, so the kernel inherits PIPELINE_PROFILE (and PIPELINE_ROOT) from this process; the profile
in effect is printed and stored in each notebook's metadata (metadata.rerun).

Re-run Stages 8-14 FIRST: the notebooks only read outputs. Several notebooks still hold hand-copied display
constants (see notebooks/RERUN_EDITS.md); re-source them after a rerun that changes results.

  python notebooks/run_notebooks.py                       # all 12, in place
  python notebooks/run_notebooks.py 8 9 10 11 12          # the Stage 8-12 notebooks
  PIPELINE_PROFILE=config/profiles/refs4_14.yaml python notebooks/run_notebooks.py 10 --out-dir tmp/nb_refs4
  python notebooks/run_notebooks.py --dry-run

CPU only; minutes per notebook. On a shared cluster run it inside a batch job, not on a login node.
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("numbers", nargs="*", type=int, help="notebook numbers N of pipelineN_analysis.ipynb (default all)")
    ap.add_argument("--out-dir", default=None, help="write executed copies here instead of in place")
    ap.add_argument("--kernel", default="python3",
                    help="kernelspec name (default python3 = this interpreter's own kernel)")
    ap.add_argument("--timeout", type=int, default=1800, help="per-cell timeout, seconds")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failing notebook")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    nums = a.numbers or sorted(int(p.stem.split("_")[0].replace("pipeline", ""))
                               for p in NB_DIR.glob("pipeline*_analysis.ipynb"))
    paths = [NB_DIR / f"pipeline{n}_analysis.ipynb" for n in nums]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        sys.exit(f"not found: {missing}")
    profile = os.environ.get("PIPELINE_PROFILE", "")
    print(f"interpreter: {sys.executable}\nkernel: {a.kernel}\n"
          f"PIPELINE_PROFILE: {profile or '(none: config/pipeline_config.yaml)'}")
    for p in paths:
        print(f"  {p.relative_to(ROOT)}")
    if a.dry_run:
        return

    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    tmp = ROOT / "tmp" / "run_notebooks"
    (tmp / "runtime").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PIPELINE_ROOT", str(ROOT))
    os.environ.setdefault("TMPDIR", str(tmp))
    os.environ["JUPYTER_RUNTIME_DIR"] = str(tmp / "runtime")
    os.environ.setdefault("MPLBACKEND", "Agg")
    out_dir = Path(a.out_dir).resolve() if a.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for p in paths:
        t0 = time.time()
        nb = nbformat.read(p, as_version=4)
        client = NotebookClient(nb, kernel_name=a.kernel, timeout=a.timeout,
                                resources={"metadata": {"path": str(NB_DIR)}})
        try:
            client.execute()
            ok = True
        except CellExecutionError as e:
            ok = False
            print(f"FAILED {p.name}: {str(e).splitlines()[0] if str(e) else e}")
        nb.metadata["rerun"] = {"pipeline_profile": profile, "ok": ok,
                                "executed": time.strftime("%Y-%m-%dT%H:%M:%S")}
        dest = (out_dir / p.name) if out_dir else p
        nbformat.write(nb, dest)
        print(f"{'OK    ' if ok else 'FAILED'} {p.name} ({time.time() - t0:.0f}s) -> {dest}")
        if not ok:
            failed.append(p.name)
            if not a.keep_going:
                break
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
