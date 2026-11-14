"""The held-out test of the model (docs/audit/remediation/STEP5_PREREG.md M2, STEP4_PREREG.md V3).

The exclusions are what make the set "held out": a series goes if it shares a GEO sample with a training
series or is its SuperSeries or SubSeries, a file goes if its sample is not exactly rat. The draw must be the
same cells every time. Nothing of this code may write into the training tree. The data tests check the record
tables against each other and against the C1 record.
"""
import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HELD = ROOT / "analysis/audit_remediation/heldout"
sys.path.insert(0, str(HELD))
import common as C   # noqa: E402

REC = ROOT / "docs/audit/remediation/record/step4_heldout"


def _soft(gsm=(), relations=(), organisms=None):
    return {"title": "", "gsm": list(gsm), "relations": list(relations), "organisms": organisms or {},
            "sample_title": {}, "source_name": {}, "characteristics": {}}


def test_series_sharing_a_sample_with_training_is_excluded():
    soft = {"GSE1": _soft(gsm=["GSM10", "GSM11"]), "GSE2": _soft(gsm=["GSM20"]), "GSE9": _soft(gsm=["GSM10"])}
    ex, notes = C.series_exclusions(["GSE1", "GSE2"], soft, {"GSE9"}, {"GSM10": {("GSE9", "soft")}})
    assert set(ex) == {"GSE1"} and ex["GSE1"][0]["rule"] == "shares_gsm_with_training"
    assert "1 of 2 samples" in ex["GSE1"][0]["detail"] and not notes


def test_super_and_sub_series_of_training_are_excluded_from_either_side():
    soft = {
        "GSE1": _soft(relations=[("SuperSeries of", "GSE9")]),            # says so itself
        "GSE2": _soft(relations=[("SubSeries of", "GSE8")]),              # says so itself
        "GSE3": _soft(),                                                  # only the training series says so
        "GSE4": _soft(relations=[("SubSeries of", "GSE100")]),            # related to a non-training series only
        "GSE9": _soft(), "GSE8": _soft(), "GSE7": _soft(relations=[("SubSeries of", "GSE3")]),
    }
    ex, _ = C.series_exclusions(["GSE1", "GSE2", "GSE3", "GSE4"], soft, {"GSE7", "GSE8", "GSE9"}, {})
    assert {a: [r["rule"] for r in v] for a, v in ex.items()} == {
        "GSE1": ["superseries_of_training"], "GSE2": ["subseries_of_training"], "GSE3": ["superseries_of_training"]}


def test_a_sibling_under_a_shared_superseries_is_noted_not_excluded():
    # GSE304911 and the training series GSE284698 are both sub-series of GSE304998: the registered rule
    # removes the super-series, not the sibling
    soft = {"GSE304911": _soft(gsm=["GSM1"], relations=[("SubSeries of", "GSE304998")]),
            "GSE284698": _soft(gsm=["GSM2"], relations=[("SubSeries of", "GSE304998")])}
    ex, notes = C.series_exclusions(["GSE304911"], soft, {"GSE284698"}, {"GSM2": {("GSE284698", "soft")}})
    assert not ex and notes["GSE304911"][0]["rule"] == "sibling_of_training_under_superseries"


def test_a_series_without_a_soft_file_is_flagged_not_passed_silently():
    ex, notes = C.series_exclusions(["GSE5"], {"GSE5": None}, {"GSE9"}, {})
    assert not ex and notes["GSE5"][0]["rule"] == "no_soft_file"


def test_only_a_sample_whose_single_organism_is_rat_is_rat():
    assert C.sample_is_rat(["Rattus norvegicus"])
    assert not C.sample_is_rat(["Mus musculus"])
    assert not C.sample_is_rat(["Mus musculus", "Rattus norvegicus"])     # a chimera is not a rat sample
    assert not C.sample_is_rat([])


def test_parse_soft_reads_samples_relations_and_organisms(tmp_path):
    d = tmp_path / "GSE1"; d.mkdir()
    text = "\n".join([
        "^SERIES = GSE1", "!Series_title = A title", "!Series_sample_id = GSM1", "!Series_sample_id = GSM2",
        "!Series_relation = SuperSeries of: GSE2", "!Series_relation = BioProject: https://x/PRJNA1",
        "^SAMPLE = GSM1", "!Sample_title = one", "!Sample_organism_ch1 = Rattus norvegicus",
        "^SAMPLE = GSM2", "!Sample_organism_ch1 = Mus musculus", "!Sample_organism_ch1 = Rattus norvegicus"])
    with gzip.open(d / "GSE1_family.soft.gz", "wt") as f:
        f.write(text + "\n")
    s = C.parse_soft("GSE1", tmp_path)
    assert s["gsm"] == ["GSM1", "GSM2"] and s["relations"] == [("SuperSeries of", "GSE2")]
    assert s["organisms"] == {"GSM1": ["Rattus norvegicus"], "GSM2": ["Mus musculus", "Rattus norvegicus"]}
    assert C.parse_soft("GSE404", tmp_path) is None


def test_the_draw_is_fixed_capped_and_per_study():
    assert list(C.draw_cells(5, cap=10, study="GSE1")) == [0, 1, 2, 3, 4]              # under the cap: every cell
    a = C.draw_cells(10_000, cap=2000, seed=1, study="GSE1")
    assert len(a) == 2000 == len(set(a.tolist())) and a.min() >= 0 and a.max() < 10_000
    assert np.array_equal(a, np.sort(a)) and np.array_equal(a, C.draw_cells(10_000, cap=2000, seed=1, study="GSE1"))
    assert not np.array_equal(a, C.draw_cells(10_000, cap=2000, seed=1, study="GSE2"))
    assert not np.array_equal(a, C.draw_cells(10_000, cap=2000, seed=2, study="GSE1"))
    assert C.CAP_PER_STUDY == 2000                                                     # the registered cap


def test_feature_names_tell_the_genome():
    rat = [f"ENSRNOG{i:011d}" for i in range(100)]
    mouse = [f"ENSMUSG{i:011d}" for i in range(100)]
    assert C.feature_species(rat, rat)[0] == "rat"
    assert C.feature_species(mouse, mouse)[0] == "mouse"
    assert C.feature_species(rat + mouse, rat + mouse)[0] == "mixed"                   # a two-genome reference
    assert C.feature_species([f"GRCh38_ENSG{i:011d}" for i in range(50)], ["ACTB"] * 50)[0] == "human"
    sym_rat = ["Actb", "Gapdh", "LOC100911034", "RGD1560248", "AABR07000398.1"] * 20
    sym_mouse = ["Actb", "Gapdh", "Gm12345", "1700001A01Rik", "Xkr4"] * 20
    assert C.feature_species(sym_rat, sym_rat)[0] == "rat"
    assert C.feature_species(sym_mouse, sym_mouse)[0] == "mouse"
    assert C.feature_species(["ACTB", "GAPDH", "TP53"] * 30, ["ACTB", "GAPDH", "TP53"] * 30)[0] == "human"
    assert C.feature_species(["Actb", "Gapdh"] * 30, ["Actb", "Gapdh"] * 30)[0] == "unknown"   # says so, no guess


def test_csv_gene_axis_is_found_when_cells_are_the_columns(tmp_path):
    p = tmp_path / "GSE1_counts.csv"
    p.write_text("feature,LG08_AAACCCAAGGTACCTT,LG08_AAACCCAAGTCCTGCG\nUlbp1,0,1\nLOC100911034,2,0\nRGD1560248,0,0\n")
    ids, _ = C.source_feature_names(p, "csv")
    assert ids == ["Ulbp1", "LOC100911034", "RGD1560248"]
    q = tmp_path / "GSE1_cells_by_genes.csv"
    q.write_text("cell,Ulbp1,LOC100911034\nAAACCCAAGGTACCTT-1,0,1\nAAACCCAAGTCCTGCG-1,2,0\n")
    assert C.source_feature_names(q, "csv")[0] == ["Ulbp1", "LOC100911034"]


def test_tissue_groups_follow_the_sample_string_then_the_study_string():
    g = C.tissue_group
    assert g("Frozen postmortem nucleus accumbens") == "brain: nucleus accumbens / striatum"
    assert g("frozen postmortem brain", "nucleus accumbens") == "brain: nucleus accumbens / striatum"   # refined
    assert g("brain", "heart") == "brain: other or unspecified"            # a non-brain study string does not win
    assert g("SARS-Cov-2", "spleen") == "spleen"                           # a mis-parsed GEO field falls through
    assert g("Optic Nerve Head") == "eye" and g("Sciatic Nerve") == "peripheral nerve"
    assert g("L4-L6 DRG") == "dorsal root ganglion" and g("Cardiac") == "heart"
    assert g("", "") == "other / unknown" and g(None) == "other / unknown"


@pytest.mark.parametrize("sub", ["data/training/x.h5ad", "data/catalog/x", "data/models/m", "data/results/r",
                                 "data/deconvolution/d", "reports/r.tsv"])
def test_nothing_is_written_into_the_training_tree(sub):
    with pytest.raises(RuntimeError):
        C.out_path(ROOT / sub)
    assert C.out_path(C.HELD / "qc_h5ad" / "x.h5ad")
    for d in (C.QC_DIR, C.QC_REPORTS, C.TOK_DIR, C.EVAL_DIR, C.REC):
        assert C.out_path(d)


def test_the_evaluator_settings_are_the_c1_c2_settings():
    """06_eval.py must default to eval_mlm.py's seed and batch size (the C1/C2 runs passed neither)."""
    import ast
    def defaults(path):
        out = {}
        for node in ast.walk(ast.parse(Path(path).read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" and node.args:
                for kw in node.keywords:
                    if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                        out[node.args[0].value] = kw.value.value
        return out
    mine, theirs = defaults(HELD / "06_eval.py"), defaults(ROOT / "finetune/genecompass/eval_mlm.py")
    assert mine["--seed"] == theirs["--seed"] == 20261008
    assert mine["--batch-size"] == theirs["--batch-size"] == 8


# ----------------------------------------------------------------------------------------------------------
# data tests: the record tables agree with each other and with the C1 record
# ----------------------------------------------------------------------------------------------------------
def _rows(name, require):
    return C.read_tsv(require(REC / name))


@pytest.mark.data
def test_no_heldout_series_is_in_the_corpus_plan_or_shares_a_sample_with_it(require):
    plan = C.read_tsv(require(C.PLAN))
    kept = {r["series"] for r in _rows("file_plan.tsv", require) if r["action"] == "keep"}
    assert kept and not kept & {r["study"] for r in plan}
    train = C.training_gsms(plan)
    for a in sorted(kept):
        s = C.parse_soft(a)
        assert s is not None and not set(s["gsm"]) & set(train), a
        assert not {o for _, o in s["relations"]} & {r["study"] for r in plan}, a


@pytest.mark.data
def test_every_kept_file_with_a_gsm_is_a_rat_sample(require):
    for r in _rows("file_plan.tsv", require):
        if r["action"] == "keep":
            assert r["geo_organism"] in ("", C.RAT) and r["features_annotation"] not in ("mouse", "human", "mixed"), r
            assert r["reverse_normalized"] == "False", r


@pytest.mark.data
def test_the_draw_table_matches_the_plan_and_the_cap(require):
    kept = {}
    for r in _rows("file_plan.tsv", require):
        if r["action"] == "keep":
            kept[r["series"]] = kept.get(r["series"], 0) + int(r["cells_qc"])
    draw = {r["study"]: r for r in _rows("eval_draw.tsv", require)}
    assert set(draw) == set(kept)
    for s, r in draw.items():
        assert int(r["cells_tokenized"]) == kept[s]
        assert int(r["cells_drawn"]) == min(kept[s], C.CAP_PER_STUDY) and int(r["seed"]) == C.DRAW_SEED


@pytest.mark.data
def test_the_in_sample_probe_loss_is_the_recorded_c1_loss(require):
    rec = json.loads(require(ROOT / "data/results/retrain_v2/mlm/v2.json").read_text())["species"]["rat"]
    mine = json.loads(require(C.EVAL_DIR / "v2_probe.json").read_text())["species"]["rat"]
    assert mine["cells"] == rec["cells"] and mine["masked_tokens"] == rec["masked_tokens"]
    assert mine["id_loss"] == pytest.approx(rec["id_loss"], rel=1e-6)
    assert mine["value_loss"] == pytest.approx(rec["value_loss"], rel=1e-5)


@pytest.mark.data
def test_the_registered_reading_follows_from_the_loss_table(require):
    read = {r["quantity"]: r["value"] for r in _rows("registered_reading.tsv", require)}
    ratio = float(read["ratio_v2_heldout_over_v2_probe_id_loss"])
    assert ratio == pytest.approx(float(read["v2_heldout_id_loss"]) / float(read["v2_probe_id_loss"]), rel=1e-9)
    expect = "Generalises" if ratio <= 1.10 else "Partly" if ratio <= 1.25 else "Does not"
    assert read["registered_reading"] == expect


# ----------------------------------------------------------------------------------------------------------
# 06_eval.py against the evaluator it wraps, on a stand-in model (offline, CPU, a few seconds)
# ----------------------------------------------------------------------------------------------------------
def _script(name):
    spec = importlib.util.spec_from_file_location(name[:-3].replace(".", "_"), HELD / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _toy_set(path, species=(0, 1, 2), cells=20, length=32):
    import datasets
    rng = np.random.default_rng(0)
    rows = {k: [] for k in ("input_ids", "values", "length", "species", "study_id", "sample_id", "cell_index")}
    for s in species:
        for i in range(cells):
            n = int(rng.integers(8, length + 1))
            ids = rng.integers(2, 50000, size=n).tolist() + [0] * (length - n)
            vals = np.sort(rng.random(n))[::-1].astype(np.float32).tolist() + [0.0] * (length - n)
            rows["input_ids"].append(ids); rows["values"].append(vals); rows["length"].append([n])
            rows["species"].append([s]); rows["study_id"].append(f"GSE{s}{i % 3}")
            rows["sample_id"].append(f"GSE{s}{i % 3}_sample0"); rows["cell_index"].append(i)
    datasets.Dataset.from_dict(rows).save_to_disk(str(path))
    return path


def _fake_model():
    import torch
    from types import SimpleNamespace

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.emb = torch.nn.Embedding(55275, 8)
            self.cls = torch.nn.Linear(8, 55275)
            self.cls4value = torch.nn.Linear(8, 1)

        def forward(self, input_ids, values, attention_mask, species, labels, labels_values, return_dict=True):
            h = self.emb(input_ids) + values.unsqueeze(-1)
            logits = self.cls(h)
            pv = self.cls4value(h).squeeze(-1)
            m = labels != -100
            return SimpleNamespace(
                id_loss=torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1)),
                value_loss=torch.nn.functional.mse_loss(pv[m], labels_values[m]), logits=logits)
    return Fake()


def test_per_cell_wrapper_gives_the_evaluators_rat_numbers(tmp_path, monkeypatch):
    """Same cells, same model: eval_mlm.py's own `eval --by-tier` and 06_eval.py must print the same rat block,
    and the per-cell sums must add up to it."""
    pytest.importorskip("torch"); pytest.importorskip("datasets")
    ev = _script("06_eval.py")
    E = ev.E
    fake = _fake_model()
    monkeypatch.setattr(E, "load_model", lambda model, kind, device: (fake.eval().to(device), 1.0, []))
    monkeypatch.setattr(E, "load_tiers", lambda: {t: ("T1" if t % 2 else "T4") for t in range(2, 40000)})
    ds = _toy_set(tmp_path / "set")
    from types import SimpleNamespace
    E.evaluate(SimpleNamespace(model="m", kind="rat", probe=str(ds), out=str(tmp_path / "ref"), batch_size=8,
                               by_tier=True, seed=20261008))
    ref = json.loads((tmp_path / "ref.json").read_text())["species"]["rat"]
    monkeypatch.setattr(sys, "argv", ["06_eval.py", "--model", "m", "--set", str(ds), "--out", str(tmp_path / "mine")])
    ev.main()
    mine = json.loads((tmp_path / "mine.json").read_text())["species"]["rat"]
    assert mine["cells"] == ref["cells"] == 20 and mine["masked_tokens"] == ref["masked_tokens"]
    for k in ("id_loss", "value_loss", "top1_acc"):
        assert mine[k] == pytest.approx(ref[k], rel=1e-9), k
    assert set(mine["by_tier"]) == set(ref["by_tier"])
    for t, r in ref["by_tier"].items():
        assert mine["by_tier"][t]["masked_tokens"] == r["masked_tokens"]
        assert mine["by_tier"][t]["id_loss"] == pytest.approx(r["id_loss"], rel=1e-5)
        assert mine["by_tier"][t]["top1_acc"] == pytest.approx(r["top1_acc"], rel=1e-9)
    z = np.load(tmp_path / "mine.cells.npz", allow_pickle=False)
    assert len(z["row"]) == 20 and z["n_masked"].sum() == ref["masked_tokens"]
    assert z["ce_sum"].sum() / z["n_masked"].sum() == pytest.approx(ref["id_loss"], rel=1e-5)
    assert z["se_sum"].sum() / z["n_masked"].sum() == pytest.approx(ref["value_loss"], rel=1e-4)
    assert z["n_hit"].sum() / z["n_masked"].sum() == pytest.approx(ref["top1_acc"], rel=1e-9)
    assert np.array_equal(z["tier_n"].sum(1), z["n_masked"])
    assert sorted(set(z["study_id"].tolist())) == ["GSE20", "GSE21", "GSE22"]           # the rat rows only


def test_the_unchanged_evaluator_cannot_score_a_rat_only_set(tmp_path, monkeypatch):
    """Why 06_eval.py exists: eval_mlm.py divides by the number of masked human tokens."""
    pytest.importorskip("torch"); pytest.importorskip("datasets")
    ev = _script("06_eval.py")
    fake = _fake_model()
    monkeypatch.setattr(ev.E, "load_model", lambda model, kind, device: (fake.eval().to(device), 1.0, []))
    ds = _toy_set(tmp_path / "ratonly", species=(2,), cells=8)
    from types import SimpleNamespace
    with pytest.raises(ZeroDivisionError):
        ev.E.evaluate(SimpleNamespace(model="m", kind="rat", probe=str(ds), out=str(tmp_path / "x"), batch_size=8,
                                      by_tier=False, seed=20261008))
    monkeypatch.setattr(sys, "argv", ["06_eval.py", "--model", "m", "--set", str(ds), "--out", str(tmp_path / "y")])
    ev.main()
    assert json.loads((tmp_path / "y.json").read_text())["species"]["rat"]["cells"] == 8
