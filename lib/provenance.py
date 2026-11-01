"""provenance.py -- write a small JSON record of how an output was produced.

One call next to an output directory records the git commit (plus any modified tracked
files), the command line, the time, the host, the run parameters, and the size and mtime of
each input file (optionally its md5, for files under 2 GB) and, if given, the loaded config, so a
result can always be traced to the code and inputs that made it.
deconvolution/R/provenance.R is the R twin and writes the same fields.

It never raises: a failure only prints a warning, because provenance must not fail a run.

    from provenance import write_provenance
    write_provenance(out_dir, inputs=[pheno_tsv, bulk_dir], params=vars(args))
    # -> <out_dir>/provenance_<script>.json
"""
import datetime
import hashlib
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


MD5_MAX_BYTES = 2 * 1024 ** 3   # md5 only files below 2 GB; bigger ones get size + mtime only


def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _stamp(path, md5=False):
    """Size and mtime of a path; with md5=True also the md5 of a regular file under 2 GB."""
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    rec = {"path": str(p), "exists": True, "bytes": st.st_size,
           "mtime": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")}
    if md5 and p.is_file():
        if st.st_size < MD5_MAX_BYTES:
            try:
                rec["md5"] = _md5(p)
            except OSError as e:
                rec["md5_error"] = str(e)
        else:
            rec["md5"] = None
            rec["md5_skipped"] = f"file >= {MD5_MAX_BYTES} bytes"
    return rec


def _config_snapshot(config):
    """A JSON-safe copy of a loaded config (drops the resolved _project_root, keeps _profile)."""
    if config is None:
        return None
    try:
        return json.loads(json.dumps({k: v for k, v in config.items() if k != "_project_root"},
                                     default=str))
    except Exception:
        return str(config)


def write_provenance(out_dir, inputs=(), params=None, name=None, md5=False, config=None,
                     md5_inputs=None):
    """Write <out_dir>/provenance_<script>.json and return its path (None on failure).

    md5=True adds the md5 of every regular input file under 2 GB (larger files and directories
    keep size + mtime only). md5_inputs, if given, is an extra list of paths that are always
    md5'd (under the same 2 GB cap) whatever ``md5`` is, e.g. the small reference files next to
    a huge corpus input. config (a loaded pipeline or training config dict) is recorded as-is.
    """
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
            "inputs": [_stamp(p, md5=md5) for p in inputs],
            "params": params or {},
        }
        if md5_inputs:
            record["inputs_md5"] = [_stamp(p, md5=True) for p in md5_inputs]
        if config is not None:
            record["config"] = _config_snapshot(config)
        path = out_dir / (name or f"provenance_{script}.json")
        path.write_text(json.dumps(record, indent=2, default=str) + "\n")
        return path
    except Exception as e:  # provenance must never break a run
        print(f"WARNING: provenance not written for {out_dir}: {e}", file=sys.stderr)
        return None
