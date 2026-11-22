"""Which fine-tuned weights Stages 9, 12 and 13 load (deconvolution/model_checkpoint.py).

Audit finding 3.1: config/profiles/april.yaml set genecompass_model_dir to April's model and left the default
config's model_checkpoint (v2) in force, so an April run would have loaded v2's weights. Every profile must now
resolve to a checkpoint inside its own model directory.
"""
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deconvolution"))
sys.path.insert(0, str(ROOT / "lib"))

import model_checkpoint as mc                      # noqa: E402
from gene_utils import _deep_merge                 # noqa: E402

PROFILES = sorted((ROOT / "config" / "profiles").glob("*.yaml"))


def _config(profile=None):
    cfg = yaml.safe_load((ROOT / "config" / "pipeline_config.yaml").read_text())
    if profile is not None:
        _deep_merge(cfg, yaml.safe_load(profile.read_text()) or {})
        cfg["_profile"] = str(profile)
    cfg["_project_root"] = ROOT
    return cfg


def _inside(child, parent):
    child, parent = Path(child), Path(parent)
    return child == parent or parent in child.parents


def test_default_checkpoint_is_inside_the_default_model_dir():
    d = _config()["deconvolution"]
    assert _inside(d["model_checkpoint"], d["genecompass_model_dir"])


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.stem)
def test_every_profile_names_one_model(profile):
    d = _config(profile)["deconvolution"]
    assert _inside(d["model_checkpoint"], d["genecompass_model_dir"]), (
        f"{profile.name}: model_checkpoint and genecompass_model_dir name different models")


def test_april_profile_names_the_april_weights():
    d = _config(ROOT / "config" / "profiles" / "april.yaml")["deconvolution"]
    assert "rat_genecompass_finetuned/" in d["model_checkpoint"]
    assert "finetuned_v2" not in d["model_checkpoint"]


def test_a_checkpoint_outside_the_model_dir_is_refused(tmp_path):
    cfg = {"_project_root": tmp_path,
           "deconvolution": {"genecompass_model_dir": "april", "model_checkpoint": "v2/models"}}
    (tmp_path / "v2" / "models").mkdir(parents=True)
    (tmp_path / "v2" / "models" / "config.json").write_text("{}")
    (tmp_path / "april").mkdir()
    with pytest.raises(SystemExit, match="not inside"):
        mc.resolve_model_dir(cfg)


def test_a_checkpoint_inside_the_model_dir_is_accepted(tmp_path):
    cfg = {"_project_root": tmp_path,
           "deconvolution": {"genecompass_model_dir": "v2", "model_checkpoint": "v2/models"}}
    (tmp_path / "v2" / "models").mkdir(parents=True)
    (tmp_path / "v2" / "models" / "config.json").write_text("{}")
    assert mc.resolve_model_dir(cfg) == tmp_path / "v2" / "models"


def test_command_line_model_dir_is_not_checked_against_the_config(tmp_path):
    cfg = {"_project_root": tmp_path,
           "deconvolution": {"genecompass_model_dir": "april", "model_checkpoint": "v2/models"}}
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "config.json").write_text("{}")
    assert mc.resolve_model_dir(cfg, cli_model_dir=tmp_path / "elsewhere") == (tmp_path / "elsewhere").resolve()
