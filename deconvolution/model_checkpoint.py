"""model_checkpoint.py -- which fine-tuned GeneCompass weights Stages 9, 12 and 13 load.

Order of precedence:
  1. --model-dir on the command line (a model dir holding config.json, or a parent of checkpoint-* dirs,
     in which case the highest-numbered checkpoint is taken -- the old behaviour, kept for explicit use);
  2. deconvolution.model_checkpoint in the config: an EXPLICIT model dir (must hold config.json). This is
     the reproducible setting; the production runs used <genecompass_model_dir>/models (the final save);
  3. fallback: the highest-numbered checkpoint-* under deconvolution.genecompass_model_dir, with a WARNING,
     because "latest" silently changes if a training run adds checkpoints.
"""
import logging
import re
from pathlib import Path

_log = logging.getLogger(__name__)


def latest_checkpoint(base: Path) -> Path:
    """A model dir (has config.json) as-is; else the highest-numbered checkpoint-* beneath it (else base)."""
    base = Path(base)
    if (base / 'config.json').exists():
        return base
    cks = sorted(base.glob('checkpoint-*'), key=lambda p: int(re.sub(r'\D', '', p.name) or 0))
    return cks[-1] if cks else base


def resolve_model_dir(config: dict, cli_model_dir=None, logger=None) -> Path:
    from gene_utils import resolve_path
    log = logger or _log
    d = config['deconvolution']
    if cli_model_dir:
        return latest_checkpoint(Path(cli_model_dir).resolve())
    if d.get('model_checkpoint'):
        p = resolve_path(config, d['model_checkpoint'])
        if not (p / 'config.json').exists():
            raise SystemExit(f"deconvolution.model_checkpoint has no config.json: {p}\n"
                             f"  -> point it at a model dir (e.g. <genecompass_model_dir>/models)")
        return p
    p = latest_checkpoint(resolve_path(config, d['genecompass_model_dir']))
    log.warning(f"deconvolution.model_checkpoint is not set: using the latest checkpoint under "
                f"genecompass_model_dir ({p}). Set model_checkpoint explicitly so a rerun cannot pick "
                f"up different weights.")
    return p
