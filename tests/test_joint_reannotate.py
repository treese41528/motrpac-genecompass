"""Joint re-annotation: the panel-scoring rule of REFERENCE_RELABEL_PREREG.md §10.3 and the review merge.

The rule is the part that decides labels, so these tests pin each branch on a toy set of clusters:
accept, near-tie, no panel, cross-compartment doublet, descent into sub-panels and stopping at the parent.
"""
import numpy as np
import pandas as pd
import pytest

import joint_reannotate as jr


def _node(genes, compartment=None, sub=None):
    n = {"compartment": compartment, "genes": genes, "ids": genes, "found": genes, "missing": []}
    if sub:
        n["sub"] = sub
    return n


TREE = {
    "Epi": _node(["e1", "e2"], "epithelial"),
    "Imm": _node(["i1", "i2"], "immune", sub={
        "T": _node(["t1", "t2"], sub={"CD4": _node(["c4", "c4b"]), "CD8": _node(["c8", "c8b"])}),
        "B": _node(["b1", "b2"]),
    }),
    "Endo": _node(["v1", "v2"], "vascular"),
}
GENES = ["e1", "e2", "i1", "i2", "t1", "t2", "c4", "c4b", "c8", "c8b", "b1", "b2", "v1", "v2"]


def _means(rows):
    return pd.DataFrame(rows, columns=GENES).rename(index=lambda i: f"j{i}")


def _labels(rows):
    scores = pd.DataFrame(jr.score_tree(jr.scaled_means(_means(rows)), TREE))
    return jr.assign(scores, TREE)


#           e1 e2 i1 i2 t1 t2 c4 c4b c8 c8b b1 b2 v1 v2
EPI = [5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
CD8T = [0, 0, 5, 5, 5, 5, 0, 0, 5, 5, 0, 0, 0, 0]
TNOSUB = [0, 0, 5, 5, 5, 5, 3, 3, 3, 3, 0, 0, 0, 0]   # CD4 and CD8 tie -> stays "T"
BCELL = [0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0]
ENDO = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5]
DOUBLET = [4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3]  # epithelial + vascular both >= 0.5
NONE = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
TIE = [4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
CD4T = [0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0]


@pytest.fixture(scope="module")
def lab():
    return _labels([EPI, CD8T, TNOSUB, BCELL, ENDO, DOUBLET, NONE, TIE, CD4T])


def test_min_max_scaling_is_per_gene_across_clusters():
    s = jr.scaled_means(_means([EPI, ENDO, NONE]))
    assert s.loc["j0", "e1"] == 1 and s.loc["j1", "e1"] == 0 and s.loc["j2", "e1"] == pytest.approx(0.2)


def test_clear_clusters_are_accepted_and_descend(lab):
    assert lab.loc["j0", "auto_label"] == "Epi"
    assert lab.loc["j1", "auto_label"] == "CD8" and lab.loc["j1", "path"] == "Imm > T > CD8"
    assert lab.loc["j3", "auto_label"] == "B"
    assert lab.loc["j8", "auto_label"] == "CD4"
    assert (lab.loc[["j0", "j1", "j3", "j4", "j8"], "review"] == "").all()


def test_tied_children_stop_at_the_parent(lab):
    assert lab.loc["j2", "auto_label"] == "T" and lab.loc["j2", "review"] == ""


def test_cross_compartment_doublet_goes_to_review(lab):
    assert lab.loc["j5", "review"].startswith("possible doublet with Endo")
    assert lab.loc["j5", "auto_label"] == "" and lab.loc["j5", "candidate"] == "Epi"


def test_no_panel_and_near_tie_go_to_review(lab):
    assert lab.loc["j6", "review"].startswith("no panel")
    assert lab.loc["j7", "review"].startswith("near-tie")


def test_panel_with_too_few_genes_is_not_scored():
    node = {"compartment": "x", "genes": ["e1"], "ids": ["e1"], "found": ["e1"], "missing": []}
    assert jr.panel_score(jr.scaled_means(_means([EPI, ENDO])), node).isna().all()


def test_final_labels_apply_review_and_refuse_gaps(tmp_path):
    cl = pd.DataFrame({"auto_label": ["Epi", "", "", "B"], "too_small": [False, False, False, True]},
                      index=pd.Index(["j0", "j1", "j2", "j3"], name="cluster"))
    rv = tmp_path / "review.tsv"
    rv.write_text("reference\tcluster\tdecision\treason\nX\tj1\tDROP\tdoublet\n")
    with pytest.raises(SystemExit, match="j2"):
        jr.final_labels("X", cl, review=rv)
    rv.write_text(rv.read_text() + "X\tj2\tEndo\tmarkers\n")
    lab = jr.final_labels("X", cl, review=rv)
    assert lab.to_dict() == {"j0": "Epi", "j1": "DROP", "j2": "Endo", "j3": "DROP"}


def test_shared_panels_are_pulled_in_and_overridable(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("_shared:\n  A: {compartment: x, genes: [a1, a2]}\n  B: {compartment: y, genes: [b1, b2]}\n"
                 "T:\n  B: {compartment: z, genes: [z1, z2]}\n  use: [A, B]\n")
    spec = jr.load_panels("T", p)
    assert set(spec) == {"A", "B"} and spec["B"]["compartment"] == "z"


def test_every_reference_panel_file_entry_is_well_formed():
    for ref in jr.REFS:
        spec = jr.load_panels(ref)
        def walk(d, top):
            for name, n in d.items():
                assert len(n["genes"]) >= 3, (ref, name)
                if top:
                    assert n.get("compartment"), (ref, name)
                walk(n.get("sub", {}), False)
        walk(spec, True)
