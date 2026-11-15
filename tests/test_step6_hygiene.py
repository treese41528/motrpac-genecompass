"""Guards and records added in the "records, tests, hygiene" step of the audit remediation
(docs/audit/remediation/STEP6_HYGIENE.md). Each test names the finding it belongs to.

None of these changes may alter a number the pipeline produces with its defaults. What is tested here is
that the guards stop what they should, let the default path through, and that the records hold what they say.
The output-neutrality itself is checked on real data by analysis/audit_remediation/step6_neutrality_checks.py.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("finetune/genecompass", "pipeline/03_ortholog_mapping"):
    sys.path.insert(0, str(ROOT / sub))

import build_lung_pooled                         # noqa: E402
import build_ortholog_mapping as bom             # noqa: E402
import corroborate_summary                       # noqa: E402
import gene_utils                                # noqa: E402
import joint_reannotate as jr                    # noqa: E402
import perturb_cells as pc                       # noqa: E402
import project_to_atlas                          # noqa: E402
import provenance                                # noqa: E402
import run_stage6                                # noqa: E402


@pytest.fixture
def no_git(monkeypatch):
    monkeypatch.setattr(provenance, "_git", lambda *a: "")


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ provenance (cross-stage)
def test_provenance_records_the_md5_of_each_input_file_by_default(tmp_path, no_git, monkeypatch):
    monkeypatch.delenv("PROVENANCE_MD5", raising=False)
    f = tmp_path / "in.tsv"
    f.write_text("a\tb\n")
    rec = json.loads(provenance.write_provenance(tmp_path / "out", inputs=[f, tmp_path, tmp_path / "absent"]).read_text())
    file_rec, dir_rec, absent = rec["inputs"]
    assert file_rec["md5"] == md5(f) and file_rec["bytes"] == 4 and "mtime" in file_rec      # old fields kept
    assert "md5" not in dir_rec and dir_rec["exists"]                                         # a folder has none
    assert absent == {"path": str(tmp_path / "absent"), "exists": False}


def test_provenance_md5_can_be_switched_off(tmp_path, no_git, monkeypatch):
    f = tmp_path / "in.tsv"
    f.write_text("a\n")
    off = json.loads(provenance.write_provenance(tmp_path / "a", inputs=[f], md5=False).read_text())
    monkeypatch.setenv("PROVENANCE_MD5", "0")
    env_off = json.loads(provenance.write_provenance(tmp_path / "b", inputs=[f]).read_text())
    forced = json.loads(provenance.write_provenance(tmp_path / "c", inputs=[f], md5=True).read_text())
    assert "md5" not in off["inputs"][0] and "md5" not in env_off["inputs"][0]
    assert forced["inputs"][0]["md5"] == md5(f)
    assert off["inputs"][0]["bytes"] == 2


def test_provenance_does_not_hash_a_file_above_the_size_limit(tmp_path, no_git, monkeypatch):
    f = tmp_path / "big.bin"
    f.write_bytes(b"0123456789")
    monkeypatch.setattr(provenance, "MD5_MAX_BYTES", 5)
    rec = json.loads(provenance.write_provenance(tmp_path / "out", inputs=[f]).read_text())["inputs"][0]
    assert rec["md5"] is None and "md5_skipped" in rec and rec["bytes"] == 10


def test_provenance_never_raises(tmp_path, no_git, capsys):
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("x")
    assert provenance.write_provenance(blocker / "out", inputs=[]) is None
    assert "provenance not written" in capsys.readouterr().err


def test_git_state_reports_the_commit_and_the_modified_files(monkeypatch):
    def fake_git(*args):
        return "abc123\n" if args[0] == "rev-parse" else " M lib/provenance.py\nM  config/pipeline_config.yaml\n"
    monkeypatch.setattr(provenance, "_git", fake_git)
    assert provenance.git_state() == {"git_commit": "abc123",
                                      "git_modified_tracked_files": ["lib/provenance.py", "config/pipeline_config.yaml"]}
    monkeypatch.setattr(provenance, "_git", lambda *a: "")                   # git cannot be asked
    assert provenance.git_state() == {"git_commit": "", "git_modified_tracked_files": []}


# ------------------------------------------------------------------ Stage 6 overwrite guard (2.7)
def test_refuse_overwrite_stops_only_when_the_output_exists(tmp_path, capsys):
    out = tmp_path / "family_embeddings.pkl"
    gene_utils.refuse_overwrite(out)                                          # absent: nothing happens
    out.write_bytes(b"x")
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        gene_utils.refuse_overwrite(out)
    gene_utils.refuse_overwrite(out, overwrite=True)                          # asked for: allowed
    gene_utils.refuse_overwrite(out, dry_run=True)                            # dry run: reported, not fatal
    assert "a real run would stop here" in capsys.readouterr().out
    assert out.read_bytes() == b"x"


def stage6_config(tmp_path, promoter=True, grn=True):
    pk = tmp_path / "prior_knowledge"
    pk.mkdir(exist_ok=True)
    return {"_project_root": tmp_path, "paths": {"prior_knowledge_dir": "prior_knowledge"},
            "prior_knowledge": {"coexp": {}, "gene_family": {}, "promoter": {"enabled": promoter},
                                "grn": {"enabled": grn}}}, pk


def test_stage6_lists_the_outputs_a_run_would_overwrite(tmp_path):
    cfg, pk = stage6_config(tmp_path, promoter=False)
    for name in ("coexp_embeddings.pkl", "family_embeddings.pkl", "promoter_embeddings.pkl", "grn_embeddings.pkl"):
        (pk / name).write_bytes(b"x")
    assert [p.name for p in run_stage6.existing_outputs(cfg, 1)] == ["coexp_embeddings.pkl", "family_embeddings.pkl",
                                                                     "grn_embeddings.pkl"]      # promoter disabled
    assert [p.name for p in run_stage6.existing_outputs(cfg, 4)] == ["grn_embeddings.pkl"]
    (pk / "grn_embeddings.pkl").unlink()
    assert run_stage6.existing_outputs(cfg, 4) == []


@pytest.fixture
def stage6(monkeypatch, tmp_path):
    """run_stage6.main on a scratch prior-knowledge folder, with the steps recorded instead of run."""
    cfg, pk = stage6_config(tmp_path)
    ran = []
    monkeypatch.setattr(run_stage6, "load_config", lambda *a: cfg)
    monkeypatch.setattr(run_stage6, "validate_config", lambda c: True)
    monkeypatch.setattr(run_stage6, "validate_inputs", lambda c, s: True)
    monkeypatch.setattr(run_stage6, "run_step", lambda step, config, **kw: ran.append((step["num"], kw)) or True)

    def main(*argv):
        monkeypatch.setattr(sys, "argv", ["run_stage6.py", *argv])
        run_stage6.main()
    return main, pk, ran


def test_stage6_refuses_to_run_over_an_existing_output(stage6):
    main, pk, ran = stage6
    (pk / "family_embeddings.pkl").write_bytes(b"march")
    with pytest.raises(SystemExit) as e:
        main("--from", "2")
    assert e.value.code == 3 and ran == []
    assert (pk / "family_embeddings.pkl").read_bytes() == b"march"


def test_stage6_runs_when_nothing_would_be_overwritten_or_when_asked_to(stage6):
    main, pk, ran = stage6
    main("--from", "2")                                                       # empty folder: the default path
    assert [n for n, _ in ran] == [2, 3, 4] and not any(kw["overwrite"] for _, kw in ran)
    (pk / "family_embeddings.pkl").write_bytes(b"march")
    ran.clear()
    main("--from", "2", "--overwrite")
    assert [n for n, _ in ran] == [2, 3, 4] and all(kw["overwrite"] for _, kw in ran)


def test_stage6_start_step_after_the_existing_output_is_not_blocked(stage6):
    main, pk, ran = stage6
    (pk / "family_embeddings.pkl").write_bytes(b"march")
    main("--from", "4")
    assert [n for n, _ in ran] == [4]


def test_stage6_dry_run_reports_and_does_not_stop(stage6):
    main, pk, ran = stage6
    (pk / "family_embeddings.pkl").write_bytes(b"march")
    main("--from", "2", "--dry-run")
    assert [n for n, _ in ran] == [2, 3, 4] and all(kw["dry_run"] for _, kw in ran)


def test_stage6_passes_overwrite_to_every_builder_and_test_mode_to_the_promoter_only(monkeypatch, tmp_path):
    cfg, _ = stage6_config(tmp_path)
    cmds = []
    monkeypatch.setattr(run_stage6.subprocess, "run",
                        lambda cmd, **kw: cmds.append([str(c) for c in cmd]) or type("R", (), {"returncode": 0})())
    for step in run_stage6.STEPS:
        assert run_stage6.run_step(step, cfg, test_mode=True, overwrite=True)
    assert all("--overwrite" in c for c in cmds)
    assert ["--test" in c for c in cmds] == [False, False, True, False]       # build_grn_transfer.py has no --test


def test_every_stage6_builder_accepts_overwrite_and_calls_the_guard():
    for step in run_stage6.STEPS:
        src = (run_stage6.STAGE_DIR / step["script"]).read_text()
        assert "'--overwrite'" in src, step["script"]
        assert f"refuse_overwrite(output_dir / '{step['key_output']}', args.overwrite, dry_run=args.dry_run)" in src


# ------------------------------------------------------------------ reference builders (4.10, provenance)
TYPED_IN_UNTIL_2026_11_14 = [("GSE273062", ["GSE273062_sample0", "GSE273062_sample2"]),
                             ("GSE252844", ["GSE252844_sample0"])]


def test_lung_samples_derived_from_the_yaml_equal_the_list_that_was_typed_in():
    assert build_lung_pooled.healthy_samples() == TYPED_IN_UNTIL_2026_11_14


def lung_yaml(tmp_path, studies, samples):
    f = tmp_path / "tissue_references.yaml"
    f.write_text("LUNG:\n  pooled_studies:\n" + "".join(f"    - {{study: {s}}}\n" for s in studies)
                 + f"  expect:\n    sample_ids: [{', '.join(samples)}]\n")
    return f


def test_lung_samples_follow_the_yaml_study_order(tmp_path):
    f = lung_yaml(tmp_path, ["GSE2", "GSE1"], ["GSE1_sample3", "GSE2_sample0", "GSE1_sample1"])
    assert build_lung_pooled.healthy_samples(f) == [("GSE2", ["GSE2_sample0"]), ("GSE1", ["GSE1_sample3", "GSE1_sample1"])]


def test_lung_yaml_with_an_unplaced_sample_or_an_empty_study_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="cannot be assigned"):
        build_lung_pooled.healthy_samples(lung_yaml(tmp_path, ["GSE1"], ["GSE1_sample0", "GSE9_sample0"]))
    with pytest.raises(SystemExit, match="cannot be assigned"):
        build_lung_pooled.healthy_samples(lung_yaml(tmp_path, ["GSE1", "GSE2"], ["GSE1_sample0"]))
    with pytest.raises(SystemExit, match="cannot be assigned"):                # the same study listed twice
        build_lung_pooled.healthy_samples(lung_yaml(tmp_path, ["GSE1", "GSE1"], ["GSE1_sample0"]))


def test_a_study_name_that_prefixes_another_does_not_take_its_samples(tmp_path):
    f = lung_yaml(tmp_path, ["GSE1", "GSE10"], ["GSE1_sample0", "GSE10_sample0"])
    assert build_lung_pooled.healthy_samples(f) == [("GSE1", ["GSE1_sample0"]), ("GSE10", ["GSE10_sample0"])]


def test_lung_builder_has_no_default_output_folder(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["build_lung_pooled.py"])
    with pytest.raises(SystemExit) as e:
        build_lung_pooled.main()
    assert e.value.code == 2 and "--out" in capsys.readouterr().err


def test_raw_input_files_of_a_blood_and_an_external_reference(tmp_path, monkeypatch):
    d = tmp_path / "raw"
    (d / "lib").mkdir(parents=True)
    for name in ("GSM1_V1_features.tsv.gz", "GSM1_V1_barcodes.tsv.gz", "GSM1_V1_matrix.mtx.gz", "GSM1_V1_notes.txt",
                 "lib/genes.tsv", "lib/barcodes.tsv", "lib/matrix.mtx", "run.h5"):
        (d / name).write_text("x")
    monkeypatch.setattr(jr, "ROOT", tmp_path)
    monkeypatch.setattr(jr, "BLOOD_SETS", {"B": [("s1", d, "GSM1_V1_", False)]})
    monkeypatch.setattr(jr, "EXTERNAL_SETS", {"DIR": [("s2", d / "lib", "mtx_dir", True)],
                                              "PREFIX": [("s3", d / "GSM1_V1_", "mtx_dir", False)],
                                              "H5": [("s4", d / "run.h5", "h5", "symbol")]})
    monkeypatch.setattr(jr, "BULK_TISSUE", {"H5": "HEART"})
    rel = lambda ref: [str(p.relative_to(tmp_path)) for p in jr.raw_input_files(ref)]      # noqa: E731
    three = ["raw/GSM1_V1_features.tsv.gz", "raw/GSM1_V1_barcodes.tsv.gz", "raw/GSM1_V1_matrix.mtx.gz"]
    assert rel("B") == three and rel("PREFIX") == three                        # no liftover map: IDs already lifted
    assert rel("DIR") == ["raw/lib/genes.tsv", "raw/lib/barcodes.tsv", "raw/lib/matrix.mtx",
                          "deconvolution/reference/motrpac_bulk_liftover.tsv"]
    assert rel("H5") == ["raw/run.h5", "data/deconvolution/motrpac_bulk/HEART/bulk_genes.tsv"]


# ------------------------------------------------------------------ Stage 3 (1.10)
def assignment(gene, biotype, tier, token):
    return {"rat_gene": gene, "rat_symbol": gene.lower(), "biotype": biotype, "tier": tier, "token_id": token,
            "perc_id_human": "80.0", "perc_id_mouse": "85.0"}


def test_non_coding_genes_in_tiers_1_to_3_are_flagged_whatever_the_case_of_the_biotype():
    rows = [assignment("G1", "protein_coding", "T1_tri_species", 10),
            assignment("G2", "mirna", "T2a_human_one2one", 11),               # as the gene universe stores it
            assignment("G3", "miRNA", "T3a_human_multi", 50558),
            assignment("G4", "lncrna", "T4_new_token", 50559),                # tier 4: not a tier 1-3 gene
            assignment("G5", "LncRNA", "T2b_mouse_one2one", 23200)]
    _, diag = bom.compute_diagnostics(rows)
    flagged = diag["flagged_noncoding_in_tiers_1_3"]
    assert flagged["count"] == 3 and [g["rat_gene"] for g in flagged["genes"]] == ["G2", "G3", "G5"]
    assert {g["biotype"] for g in flagged["genes"]} == {"mirna", "miRNA", "LncRNA"}            # reported as stored


def test_the_flag_changes_no_tier_token_or_count():
    rows = [assignment("G1", "protein_coding", "T1_tri_species", 10), assignment("G2", "mirna", "T2a_human_one2one", 11)]
    before = [(r["rat_gene"], r["tier"], r["token_id"]) for r in rows]
    out, diag = bom.compute_diagnostics(rows)
    assert [(r["rat_gene"], r["tier"], r["token_id"]) for r in out] == before
    assert diag["biotype_tier_crosstab"] == {"protein_coding": {"T1_tri_species": 1}, "mirna": {"T2a_human_one2one": 1}}
    assert diag["tier_statistics"]["T2a_human_one2one"]["count"] == 1


# ------------------------------------------------------------------ atlas (8.6)
def test_atlas_checksum_table_has_a_row_for_every_tissue_file_the_projection_reads():
    for tissue in sorted(set(project_to_atlas.TISSUES.values())):
        row = project_to_atlas.registered_atlas_file(tissue)
        assert re.fullmatch(r"[0-9a-f]{32}", row["md5"]) and int(row["bytes"]) > 0, tissue
        assert re.fullmatch(r"[0-9a-f-]{36}", row["dataset_id"]), tissue
    assert project_to_atlas.registered_atlas_file("Kidney")["md5"] == "aad26efe8b6bcdc01fad0d00bec7c11c"


def test_an_unknown_tissue_or_a_missing_table_gives_an_empty_record(tmp_path):
    assert project_to_atlas.registered_atlas_file("Brain") == {}
    assert project_to_atlas.registered_atlas_file("Kidney", table=tmp_path / "absent.tsv") == {}


def test_the_retrieval_script_reads_the_committed_table_and_holds_no_credentials():
    src = (ROOT / "translation" / "fetch_tabula_sapiens.sh").read_text()
    assert "MD5_TSV=translation/tabula_sapiens_md5.tsv" in src and "datasets.cellxgene.cziscience.com" in src
    assert not re.search(r"token|password|api[_-]?key|authorization", src, re.I)


def test_the_sex_control_is_not_called_pre_registered():
    src = (ROOT / "translation" / "compare_transfer.py").read_text()
    assert "pre-registered control" not in src
    assert "not in a pre-registration" in src and "sex_hum_med >= 0.70" in src       # the rule itself is unchanged


# ------------------------------------------------------------------ perturbation engine (9.6)
def test_perturbation_table_gets_a_provenance_record_with_the_weights_md5(tmp_path, monkeypatch, no_git):
    model = tmp_path / "model"
    model.mkdir()
    (model / "pytorch_model.bin").write_bytes(b"weights")
    out = tmp_path / "perturb" / "mef2c.tsv"
    monkeypatch.setattr(pc, "_ROOT", ROOT)
    monkeypatch.setattr(pc, "load_token_dict", lambda p: {"ENSRNOG00000000001": 5})
    monkeypatch.setattr(pc, "load_model", lambda *a, **k: None)
    monkeypatch.setattr(pc, "load_cells", lambda *a, **k: (None, None, 3))
    monkeypatch.setattr(pc, "perturb_gene", lambda *a, **k: {"n_cells": 3, "mean_cell_shift": 0.25,
                                                             "median_cell_shift": 0.2})
    monkeypatch.setattr(sys, "argv", ["perturb_cells.py", "--model-dir", str(model), "--dataset", "unused",
                                      "--genes", "ENSRNOG00000000001", "ENSRNOG00000099999", "--out", str(out),
                                      "--device", "cpu"])
    pc.main()
    assert out.read_text().splitlines() == ["gene\tcell_type\tn_cells\tmean_cell_shift\tmedian_cell_shift\ttop_targets",
                                            "ENSRNOG00000000001\tALL\t3\t0.25\t0.2\t"]
    rec = json.loads((out.parent / "provenance_mef2c.json").read_text())
    assert rec["inputs_md5"][0]["md5"] == md5(model / "pytorch_model.bin")
    p = rec["params"]
    assert p["genes_requested"] == ["ENSRNOG00000000001", "ENSRNOG00000099999"]
    assert p["n_genes_scored"] == 1 and p["n_cells"] == 3 and p["mode"] == "delete"


# ------------------------------------------------------------------ one threshold, two languages
def test_the_de_and_the_gate_use_the_same_median_library_threshold():
    """The analysable-block rule is defined once in the gate (Python) and once as the DE's default (R). They
    must agree; this test is the link between them."""
    src = (ROOT / "deconvolution" / "R" / "run_pseudobulk_de.R").read_text()
    found = re.findall(r'MIN_MEDIAN_LIBSIZE\s*<-\s*as\.numeric\(Sys\.getenv\("DE_MIN_MEDIAN_LIBSIZE",\s*"([0-9.]+)"\)\)', src)
    assert len(found) == 1, "the R script must define the threshold exactly once"
    assert float(found[0]) == corroborate_summary.MIN_MEDIAN_LIBSIZE == 1000.0


# ------------------------------------------------------------------ Stage 13 names (kept on purpose)
def test_stage13_network_names_are_unchanged():
    import run_stage13
    assert run_stage13._safe("blood", "CD4+ T cells") == "blood_cd4pos_t_cells"
    assert run_stage13._safe("skmvl", "Vascular smooth muscle cells") == "skmvl_vascular_smooth_muscle_cells"
    assert run_stage13._safe("bat", "SMCs & Pericytes") == "bat_smcs_&_pericytes"
