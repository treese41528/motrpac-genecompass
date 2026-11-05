"""slurm_site.py -- sbatch account/partition/QOS/GRES from the per-site profile, never hardcoded.

Values come from the environment first, then deconvolution/setup/site.env (gitignored; copy from
site.env.gilbreth.example). Every knob is optional: an unset one is simply not passed to sbatch, so SLURM's
own defaults for the user apply.

  MGC_SLURM_ACCOUNT    --account       MGC_SLURM_QOS   --qos
  MGC_SLURM_PARTITION  --partition     MGC_SLURM_GRES  --gres for CPU-only jobs on clusters that require a GPU
                                                       request on every job (Gilbreth: gpu:1); GPU jobs always
                                                       request gpu:1
"""
import os
import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE_ENV = ROOT / "deconvolution" / "setup" / "site.env"
KEYS = ("MGC_SLURM_ACCOUNT", "MGC_SLURM_PARTITION", "MGC_SLURM_QOS", "MGC_SLURM_GRES")


def _parse_site_env(path=SITE_ENV):
    """KEY=value / KEY="value" lines of a shell env file (no expansion beyond quotes)."""
    vals = {}
    if not Path(path).exists():
        return vals
    for line in Path(path).read_text().splitlines():
        m = re.match(r'^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$', line)
        if not m:
            continue
        try:
            parts = shlex.split(m.group(2), comments=True)
        except ValueError:
            continue
        vals[m.group(1)] = parts[0] if parts else ""
    return vals


def site_slurm():
    """{key: value} for the four knobs (missing/empty -> absent)."""
    f = _parse_site_env()
    out = {}
    for k in KEYS:
        v = os.environ.get(k, f.get(k, ""))
        if v:
            out[k] = v
    return out


def sbatch_site_args(account=None, partition=None, qos=None, gpu=False):
    """sbatch flags from CLI overrides, else the site profile. gpu=True always requests gpu:1."""
    s = site_slurm()
    args = []
    acc = account or s.get("MGC_SLURM_ACCOUNT")
    par = partition or s.get("MGC_SLURM_PARTITION")
    q = qos or s.get("MGC_SLURM_QOS")
    if acc:
        args.append(f"--account={acc}")
    if par:
        args.append(f"--partition={par}")
    if q:
        args.append(f"--qos={q}")
    gres = "gpu:1" if gpu else s.get("MGC_SLURM_GRES")
    if gres:
        args.append(f"--gres={gres}")
    return args
