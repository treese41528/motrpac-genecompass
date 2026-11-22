#!/usr/bin/env python3
"""docs_refs5_exec_notebooks.py -- execute a worktree's analysis notebooks with the main tree as project root.

Used for the docs pass after refs5 (docs/DOCS_REFS5_UPDATE.md, section 3; jobs 12392406 and 12392439).
Same engine and metadata as notebooks/run_notebooks.py (nbclient, a kernel of this interpreter, metadata.rerun).
The only difference: the notebook files are read from a git worktree, while the kernel runs in the main tree's
notebooks/ folder, so nb_config.py (identical in both trees) resolves the project root to the main tree. In a
worktree, data/ is a symlink into the main tree, which makes notebook 8's path listing fail and puts the
worktree's path into every stored output. Executed copies go to <out dir>; nothing is written in either tree's
tracked files. Run it inside a batch job.

  python analysis/audit_remediation/docs_refs5_exec_notebooks.py <worktree> <main tree> <out dir> 8 9 10 11 12
"""
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

wt, main, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
nums = [int(x) for x in sys.argv[4:]]
out.mkdir(parents=True, exist_ok=True)
scratch = out.parent
(scratch / "runtime").mkdir(parents=True, exist_ok=True)
os.environ["PIPELINE_ROOT"] = str(main)
os.environ.setdefault("TMPDIR", str(scratch))
os.environ["JUPYTER_RUNTIME_DIR"] = str(scratch / "runtime")
os.environ.setdefault("MPLBACKEND", "Agg")
profile = os.environ.get("PIPELINE_PROFILE", "")
print(f"interpreter: {sys.executable}\nPIPELINE_PROFILE: {profile or '(none: config/pipeline_config.yaml)'}\n"
      f"notebooks from: {wt / 'notebooks'}\nkernel cwd: {main / 'notebooks'}")

failed = []
for n in nums:
    p = wt / "notebooks" / f"pipeline{n}_analysis.ipynb"
    t0 = time.time()
    nb = nbformat.read(p, as_version=4)
    client = NotebookClient(nb, kernel_name="python3", timeout=1800,
                            resources={"metadata": {"path": str(main / "notebooks")}})
    try:
        client.execute()
        ok = True
    except CellExecutionError as e:
        ok = False
        print(f"FAILED {p.name}: {e}")
    nb.metadata["rerun"] = {"pipeline_profile": profile, "ok": ok, "executed": time.strftime("%Y-%m-%dT%H:%M:%S")}
    nbformat.write(nb, out / p.name)
    print(f"{'OK    ' if ok else 'FAILED'} {p.name} ({time.time() - t0:.0f}s) -> {out / p.name}")
    if not ok:
        failed.append(p.name)
sys.exit(1 if failed else 0)
