"""Stage 13 rules registered in docs/audit/remediation/STEP3_PREREG.md section 8 (offline).

8.1  The network difference is taken on shared genes: an edge exists only where the TF and the target are in
     both arms' profiles, in the point estimate and in every bootstrap resample; sidecar tables count what was
     in one arm only; every build writes a provenance record. The decoy null goes through the same rule and
     writes each decoy's match distance; its summary gives distinct TFs beside TF x network pairs.
9.7  The rat-human self-consistency script flags and counts the TFs it scores on all targets.
5.2  Stage 13 takes the gate's hotspot set from the column is_hotspot when the gate table has one.

The model is never loaded: the builder's main() is run on a hand-made arm pair with a stand-in for the
perturbation, so the whole write path (edges, sidecars, provenance, decoy outputs) is exercised on CPU.
"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune" / "genecompass"))

import build_grn_pooled as bgp                    # noqa: E402
import compare_grn_conservation as cgc            # noqa: E402
import grn_decoy_fdr as gdf                       # noqa: E402
import run_stage13 as s13                         # noqa: E402


# ---------------------------------------------------------------- 8.1: the difference on shared genes

def test_difference_is_taken_on_the_targets_both_arms_hold():
    control = {"g1": 0.10, "g2": 0.20, "only_control": 0.90}
    trained = {"g1": 0.15, "g2": 0.05, "only_trained": 0.70}
    d = bgp.diff(control, trained)
    assert set(d) == {"g1", "g2"}                              # a target in one arm only gets no edge
    assert d["g1"] == pytest.approx(0.05) and d["g2"] == pytest.approx(-0.15)


def test_a_tf_in_one_arm_only_gets_no_edges():
    shifts = {"g1": 0.10, "g2": 0.20}
    assert bgp.diff(shifts, {}) == {}                          # TF absent from the trained profile
    assert bgp.diff({}, shifts) == {}                          # TF absent from the control profile
    assert bgp.diff({}, {}) == {}


def test_the_earlier_union_rule_is_gone():
    """Before 2026-11-14 a missing side was filled with 0, so these three pairs were 'edges' of size 0.9, 0.7, 0.3."""
    d = bgp.diff({"shared": 0.1, "only_control": 0.9}, {"shared": 0.4, "only_trained": 0.7})
    assert d == {"shared": pytest.approx(0.3)}


def test_sidecar_counts_for_one_tf():
    control = {"a": 1.0, "b": 1.0, "c": 1.0, "x": 1.0}
    trained = {"a": 1.0, "b": 1.0, "y": 1.0}
    n = bgp.arm_counts(control, trained)
    assert n == {"in_control": True, "in_trained": True, "in_both_arms": True, "n_targets_both": 2,
                 "n_targets_control_only": 2, "n_targets_trained_only": 1}
    assert n["n_targets_both"] == len(bgp.diff(control, trained))      # the edges are exactly the shared targets
    one_arm = bgp.arm_counts(control, {})
    assert (one_arm["in_control"], one_arm["in_trained"], one_arm["in_both_arms"]) == (True, False, False)
    assert (one_arm["n_targets_both"], one_arm["n_targets_control_only"], one_arm["n_targets_trained_only"]) == (0, 4, 0)


def test_network_summary_counts_tfs_and_pairs_in_one_arm_only():
    rows = [bgp.arm_counts({"a": 1, "b": 1, "x": 1}, {"a": 1, "b": 1, "y": 1}),      # in both: 2 shared, 1 + 1 one-arm
            bgp.arm_counts({"a": 1, "b": 1, "x": 1}, {}),                             # control only
            bgp.arm_counts({}, {"a": 1, "y": 1})]                                     # trained only
    overlap = {"n_genes_control": 5, "n_genes_trained": 4, "n_genes_both": 3, "n_genes_control_only": 2,
               "n_genes_trained_only": 1}
    s = bgp.network_summary(rows, overlap, n_tf_listed=10, n_edges=2, network="t_x", cell_type="X")
    assert (s["n_tf_in_both"], s["n_tf_control_only"], s["n_tf_trained_only"], s["n_tf_neither"]) == (1, 1, 1, 7)
    assert (s["n_pairs_union"], s["n_pairs_one_arm_only"]) == (4 + 3 + 2, 7)
    assert s["share_pairs_one_arm_only"] == round(7 / 9, 4)
    assert s["edge_rule"] == bgp.EDGE_RULE == "shared_genes"
    assert set(bgp.SUMMARY_COLUMNS) - {"n_control_cells", "n_trained_cells", "bootstrap", "seed"} <= set(s)


def test_sidecar_files_are_not_mistaken_for_networks(tmp_path):
    """Every consumer collects networks with the glob *_pooled.tsv; the sidecars must stay out of it."""
    out = tmp_path / "skmgn_tenocytes_pooled.tsv"
    out.write_text("tf\n")
    row = dict(bgp.arm_counts({"a": 1.0}, {"a": 1.0}), tf="T", tf_symbol="t", n_edges=1, n_boot_in_both=3,
               mean_frac_edges_in_boot=1.0)
    summ = bgp.network_summary([row], {"n_genes_control": 2, "n_genes_trained": 2, "n_genes_both": 2,
                                       "n_genes_control_only": 0, "n_genes_trained_only": 0}, 1, 1,
                               network="skmgn_tenocytes", cell_type="Tenocytes", n_control_cells=10,
                               n_trained_cells=40, bootstrap=30, seed=0)
    arms, summary = bgp.write_arm_sidecars(out, [row], summ)
    assert [p.name for p in tmp_path.glob("*_pooled.tsv")] == [out.name]
    a = pd.read_csv(arms, sep="\t")
    assert list(a.columns) == bgp.ARM_COLUMNS and bool(a.in_both_arms[0]) is True
    s = pd.read_csv(summary, sep="\t")
    assert list(s.columns) == bgp.SUMMARY_COLUMNS and len(s) == 1 and s.edge_rule[0] == "shared_genes"
    assert bgp.sidecar_paths(out)["provenance"] == str(tmp_path / "provenance_skmgn_tenocytes_pooled.json")
    assert s13.provenance_file(out) == tmp_path / "provenance_skmgn_tenocytes_pooled.json"


def test_human_builder_and_orchestrator_use_the_same_rule():
    """build_grn_human.py must take the difference from build_grn_pooled.py, and the orchestrator and the
    builder must name the rule identically (the orchestrator refuses networks built under another name)."""
    import build_grn_human as bgh
    assert bgh.diff is bgp.diff
    src = Path(bgh.__file__).read_text()
    assert ".get(k, 0" not in src and "set(tc) | set(tt)" not in src        # no private union difference
    assert s13.EDGE_RULE == bgp.EDGE_RULE


# ---------------------------------------------------------------- 8.1: the builder end to end, no model

TOK = {f"ENSRNOG{i:011d}": i for i in range(1, 9)}        # tokens 1..8
GENE = {v: k for k, v in TOK.items()}
TF_BOTH, TF_CONTROL_ONLY, TF_NEITHER = GENE[1], GENE[2], GENE[8]


def _arms():
    """Control (4 cells) holds genes 1,2,3,4,6,7; trained (6 cells) holds 1,3,4,5 and, in its first cell only, 7.
    So: gene 2 (a TF) and gene 6 are control-only, gene 5 is trained-only, gene 7 is shared in the point
    estimate but drops out of a resampled trained arm that misses the first cell."""
    def cells(n, genes, first_only=()):
        ids = torch.zeros((n, 8), dtype=torch.long)
        vals = torch.zeros((n, 8), dtype=torch.float)
        for c in range(n):
            mine = list(genes) + (list(first_only) if c == 0 else [])
            for j, g in enumerate(mine):
                ids[c, j] = g
                vals[c, j] = 1.0 + 0.5 * g + 0.2 * c          # differs by gene and by cell
        return ids, vals
    return cells(4, (1, 2, 3, 4, 6, 7)), cells(6, (1, 3, 4, 5), first_only=(7,))


def _fake_perturb(model, pid, pvl, token, device, invert):
    """Stand-in for the model: a target's shift is a fixed function of its own value and of the deleted token.
    Same contract as perturb_profile: {} when the token is not in the profile, else every other gene."""
    ids, vals = pid.tolist(), pvl.tolist()
    if token not in ids:
        return {}
    return {invert[t]: 0.001 * v * (1 + 0.1 * token) for t, v in zip(ids, vals) if t not in (0, token)}


@pytest.fixture()
def fake_build(tmp_path, monkeypatch):
    """Run build_grn_pooled.main() on the hand-made arms with the stand-in perturbation. Returns run(*extra)."""
    root = tmp_path / "root"
    (root / "data" / "training" / "ortholog_mappings").mkdir(parents=True)
    (root / "data" / "training" / "ortholog_mappings" / "rat_token_mapping.tsv").write_text(
        "rat_gene\trat_symbol\n" + "".join(f"{g}\tSym{t}\n" for g, t in TOK.items()))
    tf_file = tmp_path / "tfs.tsv"
    tf_file.write_text("gene\n" + "".join(f"{g}\n" for g in (TF_BOTH, TF_CONTROL_ONLY, TF_NEITHER)))
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "pytorch_model.bin").write_bytes(b"weights")
    monkeypatch.setattr(bgp, "_ROOT", root)
    # The provenance record asks git for the commit and the modified files; on this repository that takes
    # seconds per call and has nothing to do with what is tested here.
    import provenance
    monkeypatch.setattr(provenance, "_git", lambda *a: "")
    monkeypatch.setattr(bgp.pc, "load_token_dict", lambda *a, **k: dict(TOK))
    monkeypatch.setattr(bgp.pc, "load_model", lambda *a, **k: None)
    monkeypatch.setattr(bgp, "load_arms", lambda *a, **k: _arms())
    monkeypatch.setattr(bgp, "perturb_profile", _fake_perturb)

    def run(name, *extra):
        out = tmp_path / "grn" / f"{name}_pooled.tsv"
        monkeypatch.setattr(sys, "argv", ["build_grn_pooled.py", "--model-dir", str(model_dir), "--dataset", "unused",
                                          "--meta", str(tf_file), "--cell-type", "Hand made", "--tf-file", str(tf_file),
                                          "--out", str(out), "--bootstrap", "20", "--seed", "0", "--device", "cpu",
                                          *extra])
        bgp.main()
        return out
    return run


def test_builder_writes_edges_only_on_shared_genes(fake_build):
    out = fake_build("hand_made")
    e = pd.read_csv(out, sep="\t")
    assert list(e.columns) == ["tf", "tf_symbol", "target", "target_symbol", "delta", "boot_sd", "z"]
    # The TF in both arms has an edge to each shared target (genes 3, 4, 7) and to nothing else.
    assert set(e.tf) == {TF_BOTH}
    assert set(e.target) == {GENE[3], GENE[4], GENE[7]}
    # Hand value for gene 3: trained mean value - control mean value, times the stand-in's factor for token 1.
    (idc, vlc), (idt, vlt) = _arms()
    mean = lambda ids, vals, g: float(vals[ids == g].sum() / ids.shape[0])        # noqa: E731
    want = 0.001 * 1.1 * (mean(idt, vlt, 3) - mean(idc, vlc, 3))
    assert e.set_index("target").delta[GENE[3]] == pytest.approx(want, abs=1e-6)


def test_builder_sidecars_count_what_was_in_one_arm_only(fake_build):
    out = fake_build("hand_made")
    arms = pd.read_csv(str(out)[:-4] + "_arms.tsv", sep="\t").set_index("tf")
    assert set(arms.index) == {TF_BOTH, TF_CONTROL_ONLY}                      # the TF in neither arm has no row
    both, ctrl = arms.loc[TF_BOTH], arms.loc[TF_CONTROL_ONLY]
    assert (bool(both.in_control), bool(both.in_trained), bool(both.in_both_arms)) == (True, True, True)
    # control targets of TF 1: 2,3,4,6,7; trained targets: 3,4,5,7
    assert (both.n_targets_both, both.n_targets_control_only, both.n_targets_trained_only, both.n_edges) == (3, 2, 1, 3)
    assert (bool(ctrl.in_control), bool(ctrl.in_trained), bool(ctrl.in_both_arms)) == (True, False, False)
    assert (ctrl.n_targets_both, ctrl.n_targets_control_only, ctrl.n_targets_trained_only, ctrl.n_edges) == (0, 5, 0, 0)
    assert pd.isna(ctrl.n_boot_in_both)                                       # never bootstrapped: no edges

    s = pd.read_csv(str(out)[:-4] + "_arms_summary.tsv", sep="\t").iloc[0]
    assert s.network == "hand_made" and s.edge_rule == "shared_genes"
    assert (s.n_control_cells, s.n_trained_cells) == (4, 6)
    assert (s.n_genes_control, s.n_genes_trained, s.n_genes_both, s.n_genes_control_only, s.n_genes_trained_only) \
        == (6, 5, 4, 2, 1)
    assert (s.n_tf_listed, s.n_tf_in_both, s.n_tf_control_only, s.n_tf_trained_only, s.n_tf_neither) == (3, 1, 1, 0, 1)
    assert (s.n_edges, s.n_pairs_union, s.n_pairs_one_arm_only) == (3, 11, 8)
    assert s.share_pairs_one_arm_only == pytest.approx(8 / 11, abs=1e-4)


def test_shared_gene_rule_holds_in_every_bootstrap_resample(fake_build):
    """Gene 7 is in the trained arm through its first cell only. A resample that misses that cell has no
    edge to gene 7; the sidecar reports how often the TF's edges existed, and that must equal the count
    obtained by replaying the builder's own resampling."""
    out = fake_build("hand_made")
    rng = np.random.default_rng(0)
    with_gene7 = 0
    for _ in range(20):
        rng.integers(0, 4, 4)                                  # control draw (same order of calls as the builder)
        with_gene7 += int((rng.integers(0, 6, 6) == 0).any())  # trained draw holds the first cell?
    assert 0 < with_gene7 < 20                                 # the case is actually exercised
    row = pd.read_csv(str(out)[:-4] + "_arms.tsv", sep="\t").set_index("tf").loc[TF_BOTH]
    assert row.n_boot_in_both == 20
    assert row.mean_frac_edges_in_boot == pytest.approx((2 * 20 + with_gene7) / (3 * 20), abs=1e-4)
    e = pd.read_csv(out, sep="\t").set_index("target")
    assert (e.boot_sd > 0).all()


def test_builder_writes_a_provenance_record_with_the_weights_md5(fake_build):
    out = fake_build("hand_made")
    rec = json.loads((out.parent / "provenance_hand_made_pooled.json").read_text())
    assert rec["params"]["edge_rule"] == "shared_genes"
    assert rec["params"]["bootstrap"] == 20 and rec["params"]["cell_type"] == "Hand made"
    weights = [r for r in rec["inputs_md5"] if r["path"].endswith("pytorch_model.bin")]
    assert len(weights) == 1 and weights[0]["md5"] == hashlib.md5(b"weights").hexdigest()
    assert s13.network_edge_rule(out) == "shared_genes" and gdf.edge_rule(out) == "shared_genes"


def test_decoy_build_uses_the_same_rule_and_writes_match_distances(fake_build):
    out = fake_build("hand_made_decoy", "--decoy-seed", "0")
    e = pd.read_csv(out, sep="\t")
    assert list(e.columns)[-2:] == ["decoy_for", "match_dist"]
    m = pd.read_csv(str(out)[:-4] + "_decoy_match.tsv", sep="\t")
    # Both expressed TFs are matched (the registered matching is unchanged), to two different non-TF genes.
    assert set(m.decoy_for) == {TF_BOTH, TF_CONTROL_ONLY} and m.decoy.nunique() == 2
    assert not set(m.decoy) & {TF_BOTH, TF_CONTROL_ONLY, TF_NEITHER}
    assert np.allclose(m.match_dist, (np.log1p(m.decoy_expr) - np.log1p(m.tf_expr)).abs(), atol=1e-5)
    # A decoy has edges if and only if it is in both arms, and then only to targets in both arms.
    shared = {GENE[g] for g in (1, 3, 4, 7)}
    for r in m.itertuples():
        in_both = bool(r.decoy_in_control) and bool(r.decoy_in_trained)
        assert (r.n_edges > 0) == in_both
        assert set(e.target[e.tf == r.decoy]) == ((shared - {r.decoy}) if in_both else set())
    # The distance in the edge table is the one in the match table.
    for tf, g in e.groupby("tf"):
        assert g.match_dist.nunique() == 1
        assert g.match_dist.iloc[0] == pytest.approx(m.set_index("decoy").match_dist[tf], abs=1e-6)


# ---------------------------------------------------------------- decoy scorer (finding 9.3)

def _write_network(path, tfs, decoy=False, rule=None):
    rows = []
    for i, tf in enumerate(tfs):
        for j in range(4):
            r = {"tf": tf, "tf_symbol": tf.lower(), "target": f"t{j}", "target_symbol": f"t{j}",
                 "delta": 0.001 * (i + 1) * (j + 1), "boot_sd": 0.001, "z": float((i + 1) * (j + 1))}
            if decoy:
                r.update({"decoy_for": f"TF{i}", "match_dist": 0.01 * (i + 1)})
            rows.append(r)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    if rule:
        (path.parent / f"provenance_{path.stem}.json").write_text(json.dumps({"params": {"edge_rule": rule}}))


def test_decoy_summary_gives_distinct_tfs_beside_tf_network_pairs(tmp_path):
    tf_dir, dc_dir = tmp_path / "grn", tmp_path / "grn_decoy"
    tf_dir.mkdir(); dc_dir.mkdir()
    # TF "A" is in both networks: 5 TF x network pairs, 4 distinct TFs.
    _write_network(tf_dir / "x_pooled.tsv", ["A", "B", "C"], rule="shared_genes")
    _write_network(tf_dir / "y_pooled.tsv", ["A", "D"], rule="shared_genes")
    _write_network(dc_dir / "x_pooled.tsv", ["d1", "d2", "d3"], decoy=True, rule="shared_genes")
    _write_network(dc_dir / "y_pooled.tsv", ["d1", "d4"], decoy=True, rule="shared_genes")
    R, H, curve, tstar, skipped = gdf.score(tf_dir, dc_dir)
    assert not skipped and len(H) == 5 and H.tf.nunique() == 4
    txt = gdf.summary_text(R, H, curve, tstar)
    assert "TF x network pairs: 5 in 2 networks; distinct TFs: 4" in txt
    assert "of 5 TF x network pairs at q<0.10" in txt and "of 4 distinct TFs" in txt
    assert list(R.edge_rule) == ["shared_genes", "shared_genes"]
    assert R.match_dist_max.tolist() == [0.03, 0.02]            # read from the match_dist column of the decoy table
    assert "regulator" not in txt.lower() and "confident" not in txt.lower()


def test_decoy_scorer_refuses_a_pair_built_under_different_rules(tmp_path):
    tf_dir, dc_dir = tmp_path / "grn", tmp_path / "grn_decoy"
    tf_dir.mkdir(); dc_dir.mkdir()
    _write_network(tf_dir / "x_pooled.tsv", ["A", "B"], rule="shared_genes")
    _write_network(dc_dir / "x_pooled.tsv", ["d1", "d2"], decoy=True)            # no record: the earlier rule
    R, H, curve, tstar, skipped = gdf.score(tf_dir, dc_dir)
    assert R is None and len(skipped) == 1 and "not scored" in skipped[0]


# ---------------------------------------------------------------- self-consistency fallback (finding 9.7)

def _edges(tf, deltas, zs, human=False):
    d = pd.DataFrame({"tf": tf, "tf_symbol": tf.title(), "delta": deltas, "z": zs,
                      "t_h": [f"ENSG{i:04d}" for i in range(len(deltas))]})
    return d.drop(columns="tf_symbol") if human else d


def test_fallback_to_all_targets_is_flagged_per_tf():
    rng = np.random.default_rng(1)
    many = _edges("tfa", rng.normal(size=30), np.r_[np.full(10, 3.0), np.zeros(20)])      # 10 edges with |z| >= 2
    few = _edges("tfb", rng.normal(size=30), np.r_[np.full(3, 3.0), np.zeros(27)])        # only 3: falls back
    sel, fb = cgc.select_rat_edges(many, "z2", min_shared=8)
    assert (len(sel), fb) == (10, False)
    sel, fb = cgc.select_rat_edges(few, "z2", min_shared=8)
    assert (len(sel), fb) == (30, True)
    assert cgc.select_rat_edges(few, "all", 8)[1] is False and cgc.select_rat_edges(few, "top40", 8)[1] is False


def test_fallback_count_per_network():
    rng = np.random.default_rng(2)
    rat = pd.concat([
        _edges("tfa", rng.normal(size=30), np.r_[np.full(10, 3.0), np.zeros(20)]),        # scored on 10 selected edges
        _edges("tfb", rng.normal(size=30), np.r_[np.full(3, 3.0), np.zeros(27)]),         # fallback, scored on 30
        _edges("tfc", rng.normal(size=30), np.zeros(30)),                                 # fallback, scored on 30
        _edges("tfd", rng.normal(size=30), np.full(30, 3.0)),                             # no human rows: not scored
        _edges("tfe", rng.normal(size=5), np.zeros(5)),                                   # fallback, 5 targets: too few
    ], ignore_index=True)
    hum = pd.concat([_edges(t, rng.normal(size=30), np.zeros(30), human=True) for t in ("tfa", "tfb", "tfc", "tfe")],
                    ignore_index=True)
    rows, summ = cgc.score_network("net", rat, hum, select="z2", top_k=0, min_shared=8)
    got = {r["tf"]: (r["fallback_all_targets"], r["n_rat_selected"], r["n_shared"]) for r in rows}
    assert got == {"tfa": (False, 10, 10), "tfb": (True, 30, 30), "tfc": (True, 30, 30)}
    assert (summ["n_tf_rat"], summ["n_tf_scored"], summ["n_tf_fallback"]) == (5, 3, 2)
    assert (summ["n_tf_no_human"], summ["n_tf_lt_min_shared"], summ["n_tf_nonfinite"]) == (1, 1, 0)
    assert set(cgc.PER_TF_COLUMNS) == set(rows[0]) and set(cgc.PER_CT_COLUMNS) == set(summ)


def test_self_consistency_outputs_do_not_use_the_older_wording():
    cols = cgc.PER_TF_COLUMNS + cgc.PER_CT_COLUMNS + [cgc.OUT_SUBDIR]
    assert not [c for c in cols if "conserv" in c.lower()]
    assert "self-consistency" in cgc.__doc__.lower() and "frac_rho_ge_0.3" in cgc.PER_CT_COLUMNS


# ---------------------------------------------------------------- hotspot set and edge rule in the orchestrator

GATE_OLD = ("tissue\tcell_type\tq_sup_trained\n"
            "blood\tNeutrophils\t0.01\n"
            "lung\tGoblet cells\t0.049\n"
            "lung\tClub cells\t0.06\n"
            "bat\tTiny\t\n")
GATE_NEW = ("tissue\tcell_type\tq_sup_trained\tgate_class\tis_hotspot\n"
            "blood\tNeutrophils\t0.01\thotspot\tTrue\n"
            "lung\tGoblet cells\t0.049\tborderline\tFalse\n"       # significant at 10,000 permutations, not a hotspot
            "lung\tClub cells\t0.06\tnone\tFalse\n")


def test_gate_hotspots_use_is_hotspot_when_the_column_is_present(tmp_path):
    old, new = tmp_path / "old.tsv", tmp_path / "new.tsv"
    old.write_text(GATE_OLD); new.write_text(GATE_NEW)
    assert s13.gate_hotspots(old) == ({("BLOOD", "Neutrophils"), ("LUNG", "Goblet cells")}, "q_sup_trained < 0.05")
    assert s13.gate_hotspots(new) == ({("BLOOD", "Neutrophils")}, "is_hotspot column")


def test_network_steps_stop_when_the_de_folder_has_another_hotspot_set(tmp_path):
    gate = tmp_path / "gate.tsv"
    gate.write_text(GATE_NEW)
    de = tmp_path / "pseudobulk_de"
    de.mkdir()
    head = "tissue\tcell_type\tis_hotspot\tmean_fraction\n"
    (de / "de_hotspots.tsv").write_text(head + "BLOOD\tNeutrophils\tTRUE\t0.07\n")
    msg, note = s13.hotspot_mismatch(de, gate)
    assert msg is None and "1 hotspots" in note
    assert s13.abundant_hotspots(de, []) == [("blood", "Neutrophils", "blood_neutrophils")]
    # a DE folder made from the 10,000-permutation rule still lists the borderline block
    (de / "de_hotspots.tsv").write_text(head + "BLOOD\tNeutrophils\tTRUE\t0.07\nLUNG\tGoblet cells\tTRUE\t0.02\n")
    msg, _ = s13.hotspot_mismatch(de, gate)
    assert msg and "Goblet cells" in msg
    assert s13.hotspot_mismatch(de, tmp_path / "absent.tsv")[0] is None        # no gate table: nothing to check


def test_a_network_built_under_the_earlier_rule_is_neither_skipped_nor_reused(tmp_path, monkeypatch):
    old = tmp_path / "blood_neutrophils_pooled.tsv"
    old.write_text("tf\ttf_symbol\ttarget\ttarget_symbol\tdelta\tboot_sd\tz\n")           # no provenance record
    assert s13.network_edge_rule(old) is None
    called = []
    monkeypatch.setattr(s13.subprocess, "run", lambda *a, **k: called.append(a) or pytest.fail("must not build"))
    step = {"num": 1, "name": "rat", "desc": "", "items": [{"label": "blood/Neutrophils", "out": old, "cmd": ["x"]}]}
    assert s13.run_step(step, {}, dry_run=False, force=False) is False and not called
    # with a shared-gene record the same table counts as present
    s13.provenance_file(old).write_text(json.dumps({"params": {"edge_rule": s13.EDGE_RULE}}))
    assert s13.network_edge_rule(old) == "shared_genes"
    assert s13.run_step(step, {}, dry_run=False, force=False) is True and not called


# ---------------------------------------------------------------- 8.2: the ortholog pairs the enrichment uses

@pytest.mark.data
def test_enrichment_ortholog_pairs_are_the_stage12_map(config, require):
    """run_gene_set_enrichment.R reads the pairs from rat_token_mapping.tsv because R cannot read the pickle
    Stage 12 loads. The table's human_ortholog column must be that pickle, and the pairs it keeps
    (orth_type_human == ortholog_one2one) must be one-to-one in both directions."""
    from gene_utils import resolve_path
    d = require(resolve_path(config, config["paths"]["ortholog_dir"]))
    tm = pd.read_csv(require(d / "rat_token_mapping.tsv"), sep="\t", dtype=str, keep_default_na=False)
    r2h = pickle.load(open(require(d / "rat_to_human_mapping.pickle"), "rb"))
    base = lambda x: str(x).split(".")[0].upper()                                     # noqa: E731
    r2h = {base(k): base(v) for k, v in r2h.items()}
    table = {g: h for g, h in zip(tm.rat_gene, tm.human_ortholog) if h}
    assert table == r2h
    one = tm[tm.orth_type_human == "ortholog_one2one"]
    assert one.rat_gene.is_unique and one.human_ortholog.is_unique and (one.human_ortholog != "").all()
    assert set(one.rat_gene) <= set(r2h)
