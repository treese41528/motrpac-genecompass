"""Stage 7 (fine-tuning): what can be checked without a GPU, the model or the corpus.

The audit found no unit test for any Stage 7 code (Part 3, 7c deviation 13). Covered here:
  * the training order: one length-grouped order over the whole mix, identical on every rank, so that the
    ranks' shares cover each row exactly once (the defect fixed for v2 had each phase see a quarter of it);
  * the training mix: every source cell enters, each row points back to its source, and sorted_length.pickle
    holds each row's own length (April's held 0 for every row);
  * the checked weight load: an incomplete or altered load raises (audit finding 6.7);
  * the orchestrator: how the training YAML is flattened, what Step 1 passes to rat_model_init.py, and the
    guards added on 2026-11-14 (--force stops; the legacy T3 prior path needs an explicit flag);
  * the C1/C2 scoring rules of the retrain pre-registration and the masking of the evaluation probe.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                                   # lib.provenance, as the Stage 7 scripts import it
sys.path.insert(0, str(ROOT / "finetune" / "genecompass"))

import checked_load                                             # noqa: E402
import eval_mlm                                                 # noqa: E402
import rat_model_init                                           # noqa: E402
import run_stage7                                               # noqa: E402
import score_prereg_c1c2 as c1c2                                # noqa: E402
from patches.transformers_compat import SeededLengthGroupedSampler, declare_tied_weights   # noqa: E402

V2_YAML = ROOT / "finetune" / "genecompass" / "configs" / "rat_finetune_v2.yaml"
APRIL_YAML = ROOT / "finetune" / "genecompass" / "configs" / "rat_finetune.yaml"


# ------------------------------------------------------------------ sampler
def lengths(n=1000, seed=0):
    return np.random.default_rng(seed).integers(200, 2049, size=n).tolist()


def test_sampler_yields_every_row_exactly_once():
    lens = lengths()
    s = SeededLengthGroupedSampler(batch_size=16, lengths=lens, seed=42)
    order = list(s)
    assert len(s) == 1000 and sorted(order) == list(range(1000))


def test_sampler_order_is_the_same_for_the_same_seed_and_epoch():
    lens = lengths()
    a = list(SeededLengthGroupedSampler(16, lens, seed=42))
    b = list(SeededLengthGroupedSampler(16, lens, seed=42))       # another rank builds its own sampler
    assert a == b


def test_sampler_order_changes_with_the_epoch_and_replays_on_resume():
    lens = lengths()
    s = SeededLengthGroupedSampler(16, lens, seed=42)
    first = list(s)
    s.set_epoch(1)
    second = list(s)
    assert second != first and sorted(second) == list(range(1000))
    resumed = SeededLengthGroupedSampler(16, lens, seed=42)
    resumed.set_epoch(1)
    assert list(resumed) == second


def test_ranks_sharing_one_order_cover_the_dataset_once():
    """accelerate hands batch i of the shared order to rank i mod world_size. With the same order on every
    rank the four shares are disjoint and together hold every row."""
    lens, batch, world = lengths(n=960), 16, 4
    per_rank = []
    for rank in range(world):
        order = list(SeededLengthGroupedSampler(batch, lens, seed=42))          # built independently per rank
        batches = [order[i:i + batch] for i in range(0, len(order), batch)]
        per_rank.append([i for b in batches[rank::world] for i in b])
    assert all(len(r) == 240 for r in per_rank)
    assert sorted(i for r in per_rank for i in r) == list(range(960))


def test_sampler_groups_rows_of_similar_length():
    """Length grouping: neighbouring rows in the order are closer in length than in a plain shuffle, and the
    longest row comes first (transformers' get_length_grouped_indices)."""
    lens = lengths(n=4000)
    order = list(SeededLengthGroupedSampler(16, lens, seed=0))
    assert lens[order[0]] == max(lens)
    grouped = np.abs(np.diff(np.array(lens)[order])).mean()
    shuffled = np.abs(np.diff(np.array(lens)[np.random.default_rng(1).permutation(4000)])).mean()
    assert grouped < 0.1 * shuffled


def test_tied_weights_are_declared_for_saving():
    class Model:
        pass
    m = Model()
    declare_tied_weights(m)
    assert m._tied_weights_keys == ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias",
                                    "cls4value.predictions.decoder.bias"]


# ------------------------------------------------------------------ training mix
@pytest.fixture
def no_git(monkeypatch):
    """Provenance asks git for the commit; slow on this repository and not what is tested."""
    import lib.provenance as lp
    import provenance
    monkeypatch.setattr(lp, "_git", lambda *a: "")
    monkeypatch.setattr(provenance, "_git", lambda *a: "")


def tiny(n, first_token, rat=False, studies=None):
    """n tokenized cells: cell i holds i + 1 tokens starting at first_token + i (so every row is distinct and
    rows differ in length), in the columns the mix builder reads."""
    rows = {"input_ids": [[first_token + i + k for k in range(i + 1)] for i in range(n)],
            "values": [[1.0 + k for k in range(i + 1)] for i in range(n)],
            "length": [[i + 1] for i in range(n)]}
    if rat:
        studies = studies or ["GSE1"] * n
        rows.update({"study_id": studies, "sample_id": [f"{s}_sample0" for s in studies],
                     "gsm": [f"GSM{i}" for i in range(n)], "cell_id": [f"cell{i}" for i in range(n)],
                     "cell_index": list(range(n)), "extra_column": ["x"] * n})
    return rows


@pytest.fixture
def sources(tmp_path):
    from datasets import Dataset
    paths = {"human": tmp_path / "human", "mouse": tmp_path / "mouse", "rat": tmp_path / "rat" / "dataset"}
    data = {"human": tiny(6, 1000), "mouse": tiny(3, 2000),
            "rat": tiny(5, 3000, rat=True, studies=["GSE1", "GSE1", "GSE2", "GSE2", "GSE3"])}
    for k, p in paths.items():
        Dataset.from_dict(data[k]).save_to_disk(str(p))
    return paths, data


def build_mix(paths, out, **kw):
    import build_mixed_species_dataset as bms
    return bms.build_mixed_dataset(str(paths["rat"]), str(paths["human"]), str(paths["mouse"]), str(out), **kw)


def test_mix_holds_every_source_cell_and_each_row_points_back_to_it(sources, tmp_path, no_git):
    paths, data = sources
    mixed = build_mix(paths, tmp_path / "mix")
    name = {"human": "human", "mouse": "mouse", "rat": "rat/dataset"}
    species = {"human": 0, "mouse": 1, "rat": 2}
    seen = {k: set() for k in data}
    for row in mixed:
        k = next(k for k in data if name[k] == row["source"])
        assert row["species"] == species[k]
        assert row["input_ids"] == data[k]["input_ids"][row["source_row"]]
        assert row["length"] == data[k]["length"][row["source_row"]]
        seen[k].add(row["source_row"])
        if k == "rat":
            assert row["study_id"] == data[k]["study_id"][row["source_row"]] and row["cell_index"] == row["source_row"]
        else:
            assert (row["study_id"], row["sample_id"], row["gsm"], row["cell_id"], row["cell_index"]) == ("", "", "", "", -1)
    assert {k: len(v) for k, v in seen.items()} == {"human": 6, "mouse": 3, "rat": 5}      # all exhausted
    assert "extra_column" not in mixed.column_names


def test_sorted_length_pickle_holds_each_rows_own_length_in_dataset_order(sources, tmp_path, no_git):
    paths, _ = sources
    out = tmp_path / "mix"
    mixed = build_mix(paths, out)
    stored = pickle.load(open(out / "sorted_length.pickle", "rb"))
    assert stored == [row["length"][0] for row in mixed]
    assert len(set(stored)) > 1                       # April's file held the same value, 0, for every row


def test_mix_provenance_counts_rows_and_distinct_source_cells(sources, tmp_path, no_git):
    paths, _ = sources
    out = tmp_path / "mix"
    mixed = build_mix(paths, out, seed=7)
    files = list(out.glob("provenance_*.json"))       # named after the running script, here the test runner
    assert len(files) == 1
    rec = json.loads(files[0].read_text())
    p = rec["params"]
    assert p["n_rows"] == len(mixed) and p["seed"] == 7 and p["stopping_strategy"] == "all_exhausted"
    assert {k: v["distinct_source_cells"] for k, v in p["counts"].items()} == {"human": 6, "mouse": 3, "rat": 5}
    assert sum(v["rows"] for v in p["counts"].values()) == len(mixed)


def test_mix_is_never_written_over_an_existing_one(sources, tmp_path, no_git):
    paths, _ = sources
    build_mix(paths, tmp_path / "mix")
    with pytest.raises(FileExistsError, match="already holds a dataset"):
        build_mix(paths, tmp_path / "mix")


def test_mix_probabilities_must_sum_to_one(sources, tmp_path):
    paths, _ = sources
    with pytest.raises(AssertionError, match="sum to 1.0"):
        build_mix(paths, tmp_path / "mix", rat_prob=0.5, human_prob=0.45, mouse_prob=0.15)


def test_excluded_rat_studies_never_enter_the_mix(sources, tmp_path, no_git):
    paths, data = sources
    mixed = build_mix(paths, tmp_path / "mix", exclude_studies=("GSE2",))
    rat_rows = [r for r in mixed if r["species"] == 2]
    assert rat_rows and {r["study_id"] for r in rat_rows} == {"GSE1", "GSE3"}


def test_rat_corpus_without_source_columns_is_refused(tmp_path, sources):
    from datasets import Dataset
    paths, _ = sources
    bare = tmp_path / "bare" / "dataset"
    Dataset.from_dict(tiny(4, 3000)).save_to_disk(str(bare))
    with pytest.raises(ValueError, match="lacks source columns"):
        build_mix({**paths, "rat": bare}, tmp_path / "mix")


# ------------------------------------------------------------------ checked weight load
def linear(seed):
    torch.manual_seed(seed)
    return torch.nn.Linear(4, 3)


def test_a_complete_load_is_reported_and_verified():
    src, dst = linear(0), linear(1)
    rep = checked_load.load_state_checked(dst, src.state_dict())
    assert rep["n_model_tensors"] == rep["n_checkpoint_tensors"] == rep["n_loaded"] == 2
    assert rep["missing"] == [] and rep["unexpected"] == [] and rep["n_loaded_but_different"] == 0
    assert torch.equal(dst.weight, src.weight)


def test_a_model_tensor_missing_from_the_checkpoint_raises():
    state = linear(0).state_dict()
    del state["bias"]
    with pytest.raises(RuntimeError, match="1 model tensor\\(s\\) missing from the checkpoint \\['bias'\\]"):
        checked_load.load_state_checked(linear(1), state)


def test_a_checkpoint_tensor_the_model_does_not_use_raises():
    state = dict(linear(0).state_dict(), stray=torch.zeros(2))
    with pytest.raises(RuntimeError, match="1 checkpoint tensor\\(s\\) unused \\['stray'\\]"):
        checked_load.load_state_checked(linear(1), state)


def test_named_exceptions_are_allowed_and_still_reported():
    state = dict(linear(0).state_dict(), stray=torch.zeros(2))
    del state["bias"]
    rep = checked_load.load_state_checked(linear(1), state, allow_missing=("bias",), allow_unexpected=("stray",))
    assert rep["missing"] == ["bias"] and rep["unexpected"] == ["stray"] and rep["n_loaded"] == 1


def test_a_load_that_did_not_copy_the_values_raises():
    class Deaf(torch.nn.Linear):
        def load_state_dict(self, state, strict=True):          # accepts every key, copies nothing
            return torch.nn.modules.module._IncompatibleKeys([], [])
    with pytest.raises(RuntimeError, match="2 loaded tensor\\(s\\) differ"):
        checked_load.load_state_checked(Deaf(4, 3), linear(0).state_dict())


def test_report_only_mode_returns_the_problems_without_raising():
    state = linear(0).state_dict()
    del state["bias"]
    rep = checked_load.load_state_checked(linear(1), state, raise_on_error=False)
    assert rep["missing"] == ["bias"]


def test_no_exception_is_allowed_by_default():
    assert checked_load.ALLOW_MISSING == () and checked_load.ALLOW_UNEXPECTED == ()


# ------------------------------------------------------------------ orchestrator: config
def test_v2_training_yaml_flattens_to_the_keys_the_orchestrator_reads():
    c = run_stage7.load_config(str(V2_YAML))
    assert c["init_checkpoint"] == "data/models/rat_genecompass_init_v2"
    assert c["dataset_directory"] == "data/training/tokenized_corpus_v2/dataset"
    assert c["mixed_dataset"] == "data/training/mixed_species_dataset_v2"
    assert c["vocab_size"] == 55275 and c["num_species"] == 3
    assert c["t3_prior_source"] == "ortholog" and c["t4_zero_priors"] == ["promoter", "coexp"]
    assert c["init_seed"] == 0 and "allow_legacy_t3_priors" not in c
    assert c["_mixing"]["seed"] == 42
    assert c["_mixing"]["rat_prob"] + c["_mixing"]["human_prob"] + c["_mixing"]["mouse_prob"] == pytest.approx(1.0)
    assert c["_probe"] == {"output": "data/training/eval_probe_v2", "seed": 20261008, "n_per_species": 2000}
    assert c["output_directory"] == "data/models/rat_genecompass_finetuned_v2"
    assert c["_config_path"] == str(V2_YAML)


def test_april_yaml_names_aprils_folders_and_carries_no_seed():
    c = run_stage7.load_config(str(APRIL_YAML))
    assert c["init_checkpoint"] == "data/models/rat_genecompass_init"
    assert c["output_directory"] == "data/models/rat_genecompass_finetuned"
    assert c["init_seed"] == 0                      # the loader's default: the file has no initialization.seed
    assert c["_mixing"] == {} and c["_probe"] == {}


def test_phase_sections_do_not_leak_into_the_flat_config():
    c = run_stage7.load_config(str(V2_YAML))
    assert "freeze_encoder" not in c and "max_learning_rate" not in c


def test_legacy_pipeline_config_returns_its_finetuning_section(tmp_path):
    f = tmp_path / "pipeline_config.yaml"
    f.write_text("paths: {a: b}\nfinetuning:\n  vocab_size: 7\n")
    assert run_stage7.load_config(str(f)) == {"vocab_size": 7}


# ------------------------------------------------------------------ orchestrator: step 1 and the guards
@pytest.fixture
def step1(monkeypatch, tmp_path):
    """run_model_init with the subprocess call recorded instead of run."""
    calls = []

    def fake_run(cmd, **kw):
        calls.append([str(c) for c in cmd])
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(run_stage7.subprocess, "run", fake_run)
    base = {"init_checkpoint": str(tmp_path / "init"), "init_seed": 0}
    return calls, base


def flag_value(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_step1_passes_the_seed_and_the_prior_policy(step1):
    calls, base = step1
    assert run_stage7.run_model_init(dict(base, t3_prior_source="ortholog", t4_zero_priors=["promoter", "coexp"]))
    cmd = calls[0]
    assert cmd[1].endswith("finetune/genecompass/rat_model_init.py")
    assert flag_value(cmd, "--seed") == "0" and flag_value(cmd, "--t3-priors") == "ortholog"
    assert flag_value(cmd, "--t4-zero-priors") == "promoter,coexp"
    assert flag_value(cmd, "--output-dir") == base["init_checkpoint"]
    assert "--allow-legacy-t3-priors" not in cmd and "--overwrite" not in cmd


def test_step1_without_a_seed_reproduces_the_unseeded_build(step1):
    calls, base = step1
    run_stage7.run_model_init(dict(base, init_seed=None))
    assert "--seed" not in calls[0]


def test_step1_passes_the_legacy_flag_only_when_the_yaml_asks_for_it(step1):
    calls, base = step1
    run_stage7.run_model_init(dict(base, t3_prior_source="stage6"))
    run_stage7.run_model_init(dict(base, t3_prior_source="stage6", allow_legacy_t3_priors=True))
    assert "--allow-legacy-t3-priors" not in calls[0] and "--allow-legacy-t3-priors" in calls[1]


def test_step1_never_rebuilds_an_existing_initial_checkpoint(step1, tmp_path):
    calls, base = step1
    (tmp_path / "init").mkdir()
    (tmp_path / "init" / "pytorch_model.bin").write_bytes(b"x")
    assert run_stage7.run_model_init(base) and calls == []


def test_step1_dry_run_runs_nothing(step1):
    calls, base = step1
    assert run_stage7.run_model_init(base, dry_run=True) and calls == []


def test_force_stops_the_orchestrator_before_anything_runs(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_stage7.py", "--force"])
    monkeypatch.setattr(run_stage7, "load_config", lambda *a: pytest.fail("must stop before reading the config"))
    with pytest.raises(SystemExit) as e:
        run_stage7.main()
    assert e.value.code == 2


def test_legacy_t3_prior_path_is_refused_before_anything_is_created(tmp_path):
    out = tmp_path / "init"
    with pytest.raises(ValueError, match="allow_legacy_t3_priors"):
        rat_model_init.extend_checkpoint(tmp_path / "base", tmp_path / "tok.pickle", tmp_path / "s6",
                                         tmp_path / "hom.pickle", out, t3_prior_source="stage6")
    assert not out.exists()


def test_legacy_t3_prior_path_runs_when_asked_for(tmp_path):
    """With the flag the guard lets the call through; it then fails on the base checkpoint, which this test
    does not provide."""
    with pytest.raises(FileNotFoundError):
        rat_model_init.extend_checkpoint(tmp_path / "base", tmp_path / "tok.pickle", tmp_path / "s6",
                                         tmp_path / "hom.pickle", tmp_path / "init", t3_prior_source="stage6",
                                         allow_legacy_t3_priors=True)


def test_unknown_prior_source_and_existing_checkpoint_are_refused(tmp_path):
    with pytest.raises(ValueError, match="must be 'ortholog' or 'stage6'"):
        rat_model_init.extend_checkpoint(tmp_path / "base", "t", "s", "h", tmp_path / "init", t3_prior_source="rat")
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "pytorch_model.bin").write_bytes(b"x")
    with pytest.raises(FileExistsError, match="already holds a checkpoint"):
        rat_model_init.extend_checkpoint(tmp_path / "base", "t", "s", "h", tmp_path / "old")


# ------------------------------------------------------------------ C1 / C2 scoring
def trainer_state(tmp_path, losses, name="trainer_state.json", global_step=None):
    f = tmp_path / name
    f.write_text(json.dumps({"global_step": len(losses) if global_step is None else global_step,
                             "log_history": [{"step": i, "id_loss": a, "value_loss": b}
                                             for i, (a, b) in enumerate(losses)] + [{"step": 0, "learning_rate": 1e-4}]}))
    return f


def test_c1a_passes_when_every_step_is_logged_once_with_finite_losses(tmp_path):
    ok, final = c1c2.check_trainer_state(trainer_state(tmp_path, [(0.7, 0.01), (0.69, 0.01), (0.68, 0.009)]))
    assert ok and final == {0: (0.7, 0.01), 1: (0.69, 0.01), 2: (0.68, 0.009)}


def test_c1a_fails_on_a_non_finite_loss_or_a_missing_step(tmp_path):
    assert not c1c2.check_trainer_state(trainer_state(tmp_path, [(0.7, 0.01), (float("nan"), 0.01)]))[0]
    assert not c1c2.check_trainer_state(trainer_state(tmp_path, [(0.7, float("inf"))], name="b.json"))[0]
    assert not c1c2.check_trainer_state(trainer_state(tmp_path, [(0.7, 0.01)], name="c.json", global_step=2))[0]


def mlm_dir(tmp_path, v2=None, init=None, base=None, cells=None):
    def model(sp_losses, n_cells=2000):
        return {"emb_warmup_alpha": 1.0, "missing_keys": [],
                "species": {sp: {"cells": n_cells, "masked_tokens": 500000, "id_loss": a, "value_loss": b,
                                 "top1_acc": 0.3} for sp, (a, b) in sp_losses.items()}}
    d = tmp_path / "mlm"
    d.mkdir(exist_ok=True)
    files = {"base": model(base or {"human": (2.0, 0.010), "mouse": (2.0, 0.010)}),
             "init_v2": model(init or {"human": (2.0, 0.010), "mouse": (2.0, 0.010), "rat": (4.0, 0.030)}),
             "v2": model(v2 or {"human": (2.1, 0.0105), "mouse": (2.0, 0.010), "rat": (0.66, 0.008)},
                         n_cells=cells or 2000)}
    for k, v in files.items():
        (d / f"{k}.json").write_text(json.dumps(v))
    (d / "notes.json").write_text(json.dumps({"not": "a model"}))            # ignored: no species block
    return d


def test_c1b_and_c2_pass_when_rat_improves_and_human_mouse_stay_within_ten_percent(tmp_path):
    assert c1c2.score_mlm(mlm_dir(tmp_path)) == (True, True)


def test_c2_fails_above_ten_percent_and_passes_just_below(tmp_path):
    over = {"human": (2.21, 0.010), "mouse": (2.0, 0.010), "rat": (0.66, 0.008)}
    under = {"human": (2.19, 0.0109), "mouse": (2.0, 0.010), "rat": (0.66, 0.008)}
    assert c1c2.score_mlm(mlm_dir(tmp_path, v2=over)) == (True, False)
    assert c1c2.score_mlm(mlm_dir(tmp_path, v2=under)) == (True, True)


def test_c1b_fails_when_a_rat_loss_does_not_fall(tmp_path):
    flat = {"human": (2.0, 0.010), "mouse": (2.0, 0.010), "rat": (0.66, 0.030)}        # value loss unchanged
    assert c1c2.score_mlm(mlm_dir(tmp_path, v2=flat)) == (False, True)


def test_models_scored_on_different_cells_fail_both_criteria(tmp_path):
    assert c1c2.score_mlm(mlm_dir(tmp_path, cells=1999)) == (False, False)


def test_a_missing_model_file_stops_the_scoring(tmp_path):
    d = mlm_dir(tmp_path)
    (d / "v2.json").unlink()
    with pytest.raises(SystemExit, match="v2.json missing"):
        c1c2.score_mlm(d)


# ------------------------------------------------------------------ evaluation probe masking
def probe_batch(n=400, length=200, pad_from=150):
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(2, 5000, (n, length), generator=gen)
    ids[:, pad_from:] = eval_mlm.PAD_ID
    vals = torch.rand((n, length), generator=gen) + 0.5
    vals[:, pad_from:] = 0
    return ids, vals


def test_masking_never_touches_padding_and_labels_only_masked_positions():
    ids, vals = probe_batch()
    new_ids, new_vals, labels, label_vals = eval_mlm.mask_batch(ids, vals, torch.Generator().manual_seed(1), 5000)
    masked = labels != -100
    assert not masked[ids == eval_mlm.PAD_ID].any()
    assert torch.equal(labels[masked], ids[masked]) and torch.equal(label_vals, vals)
    assert torch.equal(new_ids[~masked], ids[~masked]) and torch.equal(new_vals[~masked], vals[~masked])


def test_masking_follows_fifteen_percent_then_eighty_ten_ten():
    ids, vals = probe_batch()
    new_ids, new_vals, labels, _ = eval_mlm.mask_batch(ids, vals, torch.Generator().manual_seed(1), 5000)
    real, masked = ids != eval_mlm.PAD_ID, labels != -100
    assert masked.sum().item() / real.sum().item() == pytest.approx(0.15, abs=0.01)
    hidden = masked & (new_vals == -1)                      # mask token or random token: the value is hidden
    as_mask = masked & (new_ids == eval_mlm.MASK_ID)
    kept = masked & (new_vals != -1)
    assert as_mask.sum().item() / masked.sum().item() == pytest.approx(0.80, abs=0.02)
    assert (hidden & ~as_mask).sum().item() / masked.sum().item() == pytest.approx(0.10, abs=0.02)
    assert kept.sum().item() / masked.sum().item() == pytest.approx(0.10, abs=0.02)
    assert torch.equal(new_ids[kept], ids[kept]) and torch.equal(new_vals[kept], vals[kept])


def test_masking_is_reproducible_from_the_generator_seed():
    ids, vals = probe_batch(n=50)
    a = eval_mlm.mask_batch(ids, vals, torch.Generator().manual_seed(5), 5000)
    b = eval_mlm.mask_batch(ids, vals, torch.Generator().manual_seed(5), 5000)
    c = eval_mlm.mask_batch(ids, vals, torch.Generator().manual_seed(6), 5000)
    assert all(torch.equal(x, y) for x, y in zip(a, b)) and not torch.equal(a[0], c[0])
