"""Shared setup: put the pipeline's script folders on sys.path so tests import the real code."""
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PIPELINE_ROOT", str(ROOT))
for sub in ("lib", "pipeline", "pipeline/01_data_harvesting", "pipeline/02_gene_universe",
            "pipeline/05_tokenization", "deconvolution", "translation"):
    sys.path.insert(0, str(ROOT / sub))


@pytest.fixture(scope="session")
def geo_harvester():
    """geo_harvester refuses to import without GEOparse, which only its metadata fetch uses.
    The file-filter and path helpers tested offline don't need it, so stand in a stub if absent."""
    try:
        import GEOparse  # noqa: F401
    except ImportError:
        sys.modules["GEOparse"] = types.ModuleType("GEOparse")
    import geo_harvester
    return geo_harvester


@pytest.fixture(scope="session")
def config():
    from gene_utils import load_config
    return load_config()


@pytest.fixture(scope="session")
def require():
    """require(path): skip a data test when the file it reads isn't on this machine."""
    def _require(path):
        if not Path(path).exists():
            pytest.skip(f"not on this machine: {path}")
        return Path(path)
    return _require
