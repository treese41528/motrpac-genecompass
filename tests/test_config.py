"""Config layering (committed -> .local.yaml -> PIPELINE_PROFILE) and the v2 / April split."""
from pathlib import Path

import pytest
import yaml

import gene_utils as gu

ROOT = Path(__file__).resolve().parents[1]


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.delenv("PIPELINE_PROFILE", raising=False)
    monkeypatch.setenv("PIPELINE_ROOT", str(tmp_path))
    _write(tmp_path / "config/pipeline_config.yaml",
           {"paths": {"a": "x", "b": "y"}, "deconvolution": {"pseudocell_target_sum": 6400, "keep": 1}})
    return tmp_path


def test_deep_merge_overrides_leaves_only():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    assert gu._deep_merge(base, {"a": {"b": 9}, "e": 4}) == {"a": {"b": 9, "c": 2}, "d": 3, "e": 4}


def test_local_override_then_profile(repo, monkeypatch):
    cfg_path = str(repo / "config/pipeline_config.yaml")
    _write(repo / "config/pipeline_config.local.yaml", {"paths": {"a": "local"}})
    _write(repo / "config/profiles/old.yaml", {"paths": {"a": "profile"},
                                               "deconvolution": {"pseudocell_target_sum": 6500}})
    cfg = gu.load_config(cfg_path)
    assert cfg["paths"] == {"a": "local", "b": "y"}
    assert "_profile" not in cfg and cfg["_project_root"] == repo.resolve()

    monkeypatch.setenv("PIPELINE_PROFILE", "config/profiles/old.yaml")    # relative to the repo root
    cfg = gu.load_config(cfg_path)
    assert cfg["paths"] == {"a": "profile", "b": "y"}                     # profile is merged last
    assert cfg["deconvolution"] == {"pseudocell_target_sum": 6500, "keep": 1}
    assert cfg["_profile"].endswith("old.yaml")


def test_missing_profile_is_an_error(repo, monkeypatch):
    monkeypatch.setenv("PIPELINE_PROFILE", "config/profiles/nope.yaml")
    with pytest.raises(FileNotFoundError):
        gu.load_config(str(repo / "config/pipeline_config.yaml"))


def test_resolve_path_is_relative_to_the_project_root(tmp_path):
    assert gu.resolve_path({"_project_root": tmp_path}, "data/x.tsv") == tmp_path / "data/x.tsv"


# ------------------------------------------------------------------ the committed files
@pytest.fixture(scope="module")
def committed():
    return (yaml.safe_load((ROOT / "config/pipeline_config.yaml").read_text()),
            yaml.safe_load((ROOT / "config/profiles/april.yaml").read_text()))


def test_v2_is_the_default_and_april_is_the_profile(committed):
    v2, april = committed
    assert v2["deconvolution"]["pseudocell_target_sum"] == 6400
    assert april["deconvolution"]["pseudocell_target_sum"] == 6500
    assert "genecompass_input_v2/" in v2["deconvolution"]["hotspots_table"]
    assert "genecompass_input/" in april["deconvolution"]["hotspots_table"]


def test_hotspot_table_sits_inside_its_input_root(committed):
    """Stage 12 mirrors the gate subfolder only when the table is under genecompass_input_dir."""
    for cfg in committed:
        d = cfg["deconvolution"]
        assert d["hotspots_table"].startswith(d["genecompass_input_dir"] + "/")


def test_every_profile_key_exists_in_the_committed_config(committed):
    """A misspelt profile key would be merged in silently and override nothing."""
    v2, april = committed
    def keys(d, prefix=""):
        for k, v in d.items():
            yield prefix + k
            if isinstance(v, dict):
                yield from keys(v, prefix + k + ".")
    assert set(keys(april)) - set(keys(v2)) == set()
