"""provenance.py -- write a small JSON record of how an output was produced.

One call next to an output directory records the git commit (plus any modified tracked
files), the command line, the time, the host, the run parameters, and the size and mtime of
each input file, so a result can always be traced to the code and inputs that made it.
deconvolution/R/provenance.R is the R twin and writes the same fields.

It never raises: a failure only prints a warning, because provenance must not fail a run.

    from provenance import write_provenance
    write_provenance(out_dir, inputs=[pheno_tsv, bulk_dir], params=vars(args))
    # -> <out_dir>/provenance_<script>.json
"""
import datetime
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _git(*args):
    try:
        out = subprocess.run(["git", "-C", str(_REPO), *args],
                             capture_output=True, text=True, timeout=30)
        return out.stdout if out.returncode == 0 else ""
    except Exception:
        return ""


def _stamp(path):
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {"path": str(p), "exists": True, "bytes": st.st_size,
            "mtime": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")}


def write_provenance(out_dir, inputs=(), params=None, name=None):
    """Write <out_dir>/provenance_<script>.json and return its path (None on failure)."""
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        script = Path(sys.argv[0]).stem or "python"
        # porcelain lines are "XY path"; keep the leading status column intact before slicing
        modified = [line[3:] for line in
                    _git("status", "--porcelain", "--untracked-files=no").splitlines() if line.strip()]
        record = {
            "script": script,
            "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": _git("rev-parse", "HEAD").strip(),
            "git_modified_tracked_files": modified,
            "command": [sys.executable, *sys.argv],
            "cwd": os.getcwd(),
            "host": socket.gethostname(),
            "inputs": [_stamp(p) for p in inputs],
            "params": params or {},
        }
        path = out_dir / (name or f"provenance_{script}.json")
        path.write_text(json.dumps(record, indent=2, default=str) + "\n")
        return path
    except Exception as e:  # provenance must never break a run
        print(f"WARNING: provenance not written for {out_dir}: {e}", file=sys.stderr)
        return None
