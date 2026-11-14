#!/usr/bin/env python3
"""step6_neutrality_checks.py -- Step 6 (records, tests, hygiene): do the Step 6 code changes leave the
pipeline's outputs unchanged?

Each check runs the changed code (the NEW tree: a snapshot of a Step 6 commit) on the data on disk, writes
into a scratch folder only, and compares the result, file by file and by md5, with either the same script at
the commit Step 6 started from (the OLD tree, 44332e6) or the production output the change must reproduce.
Nothing under data/ is written.

  refs      rebuild the LUNG, BLOOD, SPLEEN and OVARY references with the NEW builders and compare with
            data/deconvolution/references_v5/ (LUNG now reads its samples from tissue_references.yaml; all
            four builders now write a provenance record).
  stage3    rerun Stage 3 with the NEW worker into scratch and compare with data/training/ortholog_mappings/
            (the "flagged non-coding" fix must change tier_diagnostics.json in that key only).
  rin       deconvolution/R/de_technical_covariate_robustness.R, OLD against NEW, on the default tree: with
            the positional pairing allowed, with a sample key, without a key (must stop) and with a key in
            the wrong order (must stop).
  pheno     deconvolution/pheno_merge_test.py, OLD against NEW.
  augur     deconvolution/augur_prep.py, OLD against NEW (20 permutations: the check is identity of the two
            runs, not the p-values).
  atlas     translation/project_to_atlas.py score and prep (Kidney), OLD against NEW; the files on disk
            against translation/tabula_sapiens_md5.tsv.
  vst       deconvolution/R/score_z_vst.R on every purity sweep on disk, OLD against NEW and against the
            scores on disk; then NEW with SCORE_Z_VST_ROUND=1, to measure what rounding instead of
            truncation would change (the switch is off by default).
  sweepgen  deconvolution/make_purity_sweep.py, OLD against NEW, with Stage 14's command for the LUNG
            alveolar-type-II sweep (the NEW script only prints a note about a known bookkeeping defect).

  python analysis/audit_remediation/step6_neutrality_checks.py --old-tree DIR --new-tree DIR --scratch DIR \
         --record-dir docs/audit/remediation/record/step6 [--only refs stage3 ...]

Out: <record-dir>/neutrality_checks.tsv (one row per compared file or expected stop) and
<record-dir>/neutrality_checks_summary.tsv (one row per check). The trees are made by the job script
(git archive of the two commits, with data/, R_libs/ and the vendored submodules linked in).
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
ROWS = []          # check, item, reference, reference_md5, new_md5, result, detail


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def row(check, item, reference, ref_md5, new_md5, result, detail=""):
    ROWS.append({"check": check, "item": item, "compared_with": reference, "reference_md5": ref_md5,
                 "new_md5": new_md5, "result": result, "detail": detail})
    print(f"  [{check}] {item}: {result} {detail}", flush=True)


def compare_file(check, item, ref_file, new_file, reference):
    ref_file, new_file = Path(ref_file), Path(new_file)
    if not ref_file.exists() or not new_file.exists():
        row(check, item, reference, "" if not ref_file.exists() else md5(ref_file),
            "" if not new_file.exists() else md5(new_file), "MISSING",
            f"{'reference' if not ref_file.exists() else 'new'} file absent")
        return False
    a, b = md5(ref_file), md5(new_file)
    row(check, item, reference, a, b, "IDENTICAL" if a == b else "DIFFERENT")
    return a == b


def run(cmd, tree, log, env=None, expect_fail=False):
    """Run cmd with cwd = tree and PIPELINE_ROOT = PROJECT_ROOT = tree; returns (returncode, log text)."""
    e = dict(os.environ, PIPELINE_ROOT=str(tree), PROJECT_ROOT=str(tree), DECONV_PYTHON=PY,
             TMPDIR=str(Path(tree) / "tmp"))
    e.pop("PIPELINE_PROFILE", None)
    e.update(env or {})
    log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run([str(c) for c in cmd], cwd=tree, env=e, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"    ran {' '.join(str(c) for c in cmd[:4])} ... rc={rc} ({time.time() - t0:.0f}s) log={log}", flush=True)
    if rc != 0 and not expect_fail:
        print("    --- last lines ---\n" + "\n".join(log.read_text(errors="replace").splitlines()[-15:]), flush=True)
    return rc, log.read_text(errors="replace")


# ----------------------------------------------------------------------------- refs
def check_refs(a):
    S = a.scratch / "refs"
    prod = a.data / "deconvolution/references_v5"
    jl = "data/deconvolution/joint_labels/LUNG/cells_final.tsv.gz"
    builds = {
        "LUNG_native_pooled": ["deconvolution/build_lung_pooled.py", "--out", S / "LUNG_native_pooled",
                               "--joint-labels", jl],
        "BLOOD_composite": ["deconvolution/build_blood_composite.py", "build", "--out", S / "BLOOD_composite"],
        "SPLEEN_GSE306764": ["deconvolution/build_external_reference.py", "build", "--ref", "SPLEEN",
                             "--out", S / "SPLEEN_GSE306764"],
        "OVARY_GSE343214": ["deconvolution/build_external_reference.py", "build", "--ref", "OVARY",
                            "--out", S / "OVARY_GSE343214"],
    }
    for name, cmd in builds.items():
        rc, _ = run([PY] + cmd, a.new_tree, S / f"{name}.log")
        if rc != 0:
            row("refs", name, f"references_v5/{name}", "", "", "RUN FAILED", f"exit {rc}")
            continue
        for f in sorted(p.name for p in (prod / name).iterdir() if p.is_file() and not p.name.startswith("provenance")):
            same = compare_file("refs", f"{name}/{f}", prod / name / f, S / name / f, f"references_v5/{name}")
            if not same and f == "summary.txt" and (S / name / f).exists():
                # summary.txt quotes the absolute path of the liftover map, which holds the tree's own root
                old_txt = (prod / name / f).read_text().replace(str(a.main_root), "<ROOT>")
                new_txt = (S / name / f).read_text().replace(str(a.new_tree), "<ROOT>")
                row("refs", f"{name}/{f} with the repository root masked", f"references_v5/{name}", "", "",
                    "IDENTICAL" if old_txt == new_txt else "DIFFERENT",
                    "the file quotes the liftover map's absolute path; only that path differs"
                    if old_txt == new_txt else "differs beyond the path")
                if old_txt == new_txt:
                    ROWS[-2]["result"] = "PATH ONLY"
        prov = sorted(p.name for p in (S / name).glob("provenance*.json"))
        row("refs", f"{name}/provenance record", "-", "", "", "WRITTEN" if prov else "NOT WRITTEN", ", ".join(prov))
        extra = sorted(p.name for p in (S / name).iterdir()
                       if not (prod / name / p.name).exists() and not p.name.startswith("provenance"))
        if extra:
            row("refs", f"{name}/extra files", "-", "", "", "DIFFERENT", ", ".join(extra))
        big = S / name / "reference_counts.mtx"            # the matrices are 0.1-1.5 GB; keep the small files
        if big.exists():
            big.unlink()


# ----------------------------------------------------------------------------- stage3
def json_diff(x, y, path=""):
    """Paths at which two JSON values differ (dicts are descended, anything else is compared whole)."""
    if isinstance(x, dict) and isinstance(y, dict):
        out = []
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y:
                out.append(f"{path}/{k} ({'only new' if k not in x else 'only canonical'})")
            else:
                out += json_diff(x[k], y[k], f"{path}/{k}")
        return out
    return [] if x == y else [path or "/"]


def check_stage3(a):
    S = a.scratch / "stage3"
    out = S / "ortholog_mappings"
    S.mkdir(parents=True, exist_ok=True)
    prof = S / "profile.yaml"
    prof.write_text(f"paths:\n  ortholog_dir: {out}\n")
    canon = a.data / "training/ortholog_mappings"
    rc, _ = run([PY, "pipeline/03_ortholog_mapping/build_ortholog_mapping.py"], a.new_tree, S / "stage3.log",
                env={"PIPELINE_PROFILE": str(prof)})
    if rc != 0:
        row("stage3", "run", "training/ortholog_mappings", "", "", "RUN FAILED", f"exit {rc}")
        return
    # Differences this change is meant to make, and the run-specific manifest fields. Anything else is a defect.
    expected = {"tier_diagnostics.json": ("/flagged_noncoding_in_tiers_1_3",),
                "stage3_manifest.json": ("/timestamp", "/git_hash", "/git_modified_tracked_files",
                                         "/downstream/aim2_grn", "/downstream/aim3_translation"),
                "mapping_statistics.json": ("/timestamp",)}
    for f in sorted(p.name for p in out.iterdir() if p.is_file()):
        same = compare_file("stage3", f, canon / f, out / f, "training/ortholog_mappings")
        if not same and f.endswith(".json") and (canon / f).exists():
            d = json_diff(json.load(open(canon / f)), json.load(open(out / f)))
            ok = bool(d) and all(k.startswith(expected.get(f, ())) for k in d) if expected.get(f) else False
            ROWS[-1]["result"] = "EXPECTED DIFFERENCE" if ok else "DIFFERENT"
            row("stage3", f"{f}: keys that differ", "training/ortholog_mappings", "", "", "KEYS", "; ".join(d))
    diag = json.load(open(out / "tier_diagnostics.json"))["flagged_noncoding_in_tiers_1_3"]
    tiers = sorted({g["tier"] for g in diag["genes"]}); bio = sorted({g["biotype"] for g in diag["genes"]})
    row("stage3", "flagged_noncoding_in_tiers_1_3", "canonical count 0", "", "", "COUNT",
        f"count {diag['count']}; {len(diag['genes'])} listed; tiers {tiers}; biotypes {bio}")


# ----------------------------------------------------------------------------- rin / globin
def rscript(tree, script):
    return ["Rscript", "--no-init-file", "--no-site-file", Path(tree) / script]


def check_rin(a):
    S = a.scratch / "rin"
    cfg = json.loads(subprocess.run(
        [PY, "-c", "import sys, json; sys.path.insert(0, 'lib'); from gene_utils import load_config, resolve_path;"
                   "c = load_config(); d = c['deconvolution'];"
                   "print(json.dumps({k: str(resolve_path(c, d[k])) for k in "
                   "('results_dir', 'genecompass_input_dir', 'motrpac_bulk_out')}))"],
        cwd=a.new_tree, env=dict(os.environ, PIPELINE_ROOT=str(a.new_tree)), capture_output=True, text=True,
        check=True).stdout)
    results, bulk = Path(cfg["results_dir"]) / "motrpac", Path(cfg["motrpac_bulk_out"])
    hot = Path(cfg["genecompass_input_dir"]) / "pseudobulk_de" / "de_hotspots.tsv"
    with open(hot) as fh:
        tissues = sorted({r["tissue"] for r in csv.DictReader(fh, delimiter="\t")
                          if str(r["is_hotspot"]).upper() == "TRUE"})
    have_key = [t for t in tissues if (results / t / "mixture_samples.tsv").exists()]
    row("rin", "default tree", str(results), "", "", "INFO",
        f"hotspot tissues {tissues}; with a sample key on disk: {have_key or 'none'}")
    script = "deconvolution/R/de_technical_covariate_robustness.R"

    def de_dir(name):
        d = S / name; d.mkdir(parents=True, exist_ok=True)
        shutil.copy(hot, d / "de_hotspots.tsv")
        return d

    def keyed_tree(name, swap=None):
        """A results tree whose tissues link to the default pred_z and carry the sample key Stage 8 writes
        (mix<i>, line i of bulk_samples.tsv). swap = tissue whose first two vial labels are exchanged."""
        root = S / name
        for t in tissues:
            d = root / "motrpac" / t; d.mkdir(parents=True, exist_ok=True)
            if not (d / "pred_z").exists():
                os.symlink(results / t / "pred_z", d / "pred_z")
            via = [ln.strip() for ln in open(bulk / t / "bulk_samples.tsv") if ln.strip()]
            if t == swap:
                via[0], via[1] = via[1], via[0]
            with open(d / "mixture_samples.tsv", "w") as fh:
                fh.write("mix\tviallabel\n" + "".join(f"mix{i + 1}\t{v}\n" for i, v in enumerate(via)))
        return root

    out = "rin_globin_robustness.tsv"
    d_old = de_dir("old")
    rc, _ = run(rscript(a.old_tree, script), a.old_tree, S / "old.log", env={"DE_DIR": str(d_old)})
    if rc != 0:
        row("rin", "old script", "-", "", "", "RUN FAILED", f"exit {rc}")
        return
    # 1. new script, positional pairing allowed (a tree without keys)
    d = de_dir("new_allow_positional")
    rc, _ = run(rscript(a.new_tree, script), a.new_tree, S / "new_allow_positional.log",
                env={"DE_DIR": str(d), "ALLOW_POSITIONAL_SAMPLE_JOIN": "1"})
    compare_file("rin", "new script, no key on disk, ALLOW_POSITIONAL_SAMPLE_JOIN=1", d_old / out, d / out,
                 "old script (44332e6), same tree")
    # 2. new script, no key, not allowed: must stop
    d = de_dir("new_no_key")
    rc, log = run(rscript(a.new_tree, script), a.new_tree, S / "new_no_key.log", env={"DE_DIR": str(d)},
                  expect_fail=True)
    stopped = rc != 0 and "no sample key" in log and not (d / out).exists()
    if have_key:
        row("rin", "new script, default tree, no ALLOW flag", "-", "", "", "INFO",
            f"exit {rc}: the default tree has keys for {have_key}, so the stop is tested on the keyless tree below")
    else:
        row("rin", "new script, no key on disk, no ALLOW flag", "must stop", "", "",
            "STOPPED" if stopped else "DID NOT STOP", f"exit {rc}")
    # 3. new script, key present and right
    d = de_dir("new_keyed")
    rc, _ = run(rscript(a.new_tree, script), a.new_tree, S / "new_keyed.log",
                env={"DE_DIR": str(d), "CFG_RESULTS_DIR": str(keyed_tree("results_keyed"))})
    compare_file("rin", "new script, sample key present and in order", d_old / out, d / out,
                 "old script (44332e6), default tree")
    # 4. new script, key in the wrong order: must stop
    d = de_dir("new_wrong_key")
    rc, log = run(rscript(a.new_tree, script), a.new_tree, S / "new_wrong_key.log",
                  env={"DE_DIR": str(d), "CFG_RESULTS_DIR": str(keyed_tree("results_wrong_key", swap=tissues[0]))},
                  expect_fail=True)
    stopped = rc != 0 and "not identical, in order" in log and not (d / out).exists()
    row("rin", f"new script, key of {tissues[0]} with two vial labels exchanged", "must stop", "", "",
        "STOPPED" if stopped else "DID NOT STOP", f"exit {rc}")


# ----------------------------------------------------------------------------- pheno / augur
def check_pheno(a):
    S = a.scratch / "pheno"
    outs = {}
    for tag, tree in (("old", a.old_tree), ("new", a.new_tree)):
        outs[tag] = S / tag / "pheno_merge_test.tsv"
        outs[tag].parent.mkdir(parents=True, exist_ok=True)
        rc, _ = run([PY, "deconvolution/pheno_merge_test.py", "--out", outs[tag]], tree, S / f"{tag}.log")
        if rc != 0:
            row("pheno", f"{tag} run", "-", "", "", "RUN FAILED", f"exit {rc}")
            return
    compare_file("pheno", "pheno_merge_test.tsv", outs["old"], outs["new"], "old script (44332e6)")
    prov = sorted(p.name for p in (S / "new").glob("provenance*.json"))
    row("pheno", "provenance record", "-", "", "", "WRITTEN" if prov else "NOT WRITTEN", ", ".join(prov))


def check_augur(a):
    S = a.scratch / "augur"
    for tag, tree in (("old", a.old_tree), ("new", a.new_tree)):
        (S / tag).mkdir(parents=True, exist_ok=True)
        rc, _ = run([PY, "deconvolution/augur_prep.py", "--out", S / tag / "inputs", "--ctl-out",
                     S / tag / "pca_control.tsv", "--perms", "20", "--jobs", str(a.jobs)], tree, S / f"{tag}.log")
        if rc != 0:
            row("augur", f"{tag} run", "-", "", "", "RUN FAILED", f"exit {rc}")
            return
    files = ["pca_control.tsv"] + sorted(str(p.relative_to(S / "old")) for p in (S / "old" / "inputs").rglob("*.tsv"))
    n_same = 0
    for f in files:
        x, y = md5(S / "old" / f), (md5(S / "new" / f) if (S / "new" / f).exists() else "")
        n_same += x == y
        if f == "pca_control.tsv" or x != y:
            row("augur", f, "old script (44332e6)", x, y, "IDENTICAL" if x == y else "DIFFERENT")
    row("augur", f"{len(files)} files (pca_control.tsv + embed/pca/meta of every tissue)", "old script (44332e6)",
        "", "", "IDENTICAL" if n_same == len(files) else "DIFFERENT", f"{n_same} of {len(files)} identical (20 perms)")
    prov = sorted(str(p.relative_to(S / "new")) for p in (S / "new").rglob("provenance*.json"))
    row("augur", "provenance records", "-", "", "", "WRITTEN" if len(prov) == 2 else "NOT WRITTEN", ", ".join(prov))
    for tag in ("old", "new"):                                  # the exports are large; the md5s are recorded
        shutil.rmtree(S / tag / "inputs", ignore_errors=True)


# ----------------------------------------------------------------------------- atlas
def check_atlas(a):
    S = a.scratch / "atlas"
    S.mkdir(parents=True, exist_ok=True)
    atlas_dir = a.data / "external/tabula_sapiens"
    atlas_root = a.data / "deconvolution/atlas_tabula_sapiens"
    rc, log = run(["bash", "translation/fetch_tabula_sapiens.sh", "--verify"], a.new_tree, S / "verify.log",
                  env={"ATLAS_DIR": str(atlas_dir)})
    ok = [ln for ln in log.splitlines() if ln.startswith("ok ")]
    row("atlas", "files on disk against translation/tabula_sapiens_md5.tsv", str(atlas_dir), "", "",
        "IDENTICAL" if rc == 0 else "DIFFERENT", f"{len(ok)} of 11 files verify (size and md5); exit {rc}")
    human = subprocess.run(
        [PY, "-c", "import sys; sys.path.insert(0, 'lib'); from gene_utils import load_config, resolve_path;"
                   "c = load_config(); print(str(resolve_path(c, c['deconvolution']['genecompass_input_dir'])) + '_human')"],
        cwd=a.new_tree, env=dict(os.environ, PIPELINE_ROOT=str(a.new_tree)), capture_output=True, text=True,
        check=True).stdout.strip()
    for tag, tree in (("old", a.old_tree), ("new", a.new_tree)):
        rc, _ = run([PY, "translation/project_to_atlas.py", "score", "--atlas-root", atlas_root, "--human-root",
                     human, "--out", S / tag / "score"], tree, S / f"score_{tag}.log")
        if rc != 0:
            row("atlas", f"score {tag} run", "-", "", "", "RUN FAILED", f"exit {rc}")
            return
    for f in ("atlas_projection_per_type.tsv", "summary.txt"):
        compare_file("atlas", f"score/{f}", S / "old/score" / f, S / "new/score" / f, "old script (44332e6)")
        rec = Path(a.new_tree) / "translation/atlas_record_16" / f
        compare_file("atlas", f"score/{f} against the committed record", rec, S / "new/score" / f,
                     "translation/atlas_record_16")
    prov = sorted(p.name for p in (S / "new/score").glob("provenance*.json"))
    row("atlas", "score provenance record", "-", "", "", "WRITTEN" if prov else "NOT WRITTEN", ", ".join(prov))
    for tag, tree in (("old", a.old_tree), ("new", a.new_tree)):
        rc, _ = run([PY, "translation/project_to_atlas.py", "prep", "--tissue", "Kidney", "--atlas-dir", atlas_dir,
                     "--out", S / tag / "prep"], tree, S / f"prep_{tag}.log")
        if rc != 0:
            row("atlas", f"prep {tag} run", "-", "", "", "RUN FAILED", f"exit {rc}")
            return
    compare_file("atlas", "prep Kidney/prep_summary.json", S / "old/prep/Kidney/prep_summary.json",
                 S / "new/prep/Kidney/prep_summary.json", "old script (44332e6)")
    for f in sorted(p.name for p in (S / "old/prep/Kidney/dataset").iterdir() if p.suffix == ".arrow"):
        compare_file("atlas", f"prep Kidney/dataset/{f}", S / "old/prep/Kidney/dataset" / f,
                     S / "new/prep/Kidney/dataset" / f, "old script (44332e6)")
    prov = sorted(p.name for p in (S / "new/prep/Kidney").glob("provenance*.json"))
    row("atlas", "prep provenance record", "-", "", "", "WRITTEN" if prov else "NOT WRITTEN", ", ".join(prov))


# ----------------------------------------------------------------------------- purity-sweep scoring (VST)
def check_vst(a):
    """deconvolution/R/score_z_vst.R on every purity sweep on disk (config: deconvolution.validation_dir).
    The script writes into <stage>/scores, so each run gets a scratch stage folder whose mixtures/ and
    results/ are links to the sweep's own. Three runs per sweep: OLD, NEW (default) and NEW with
    SCORE_Z_VST_ROUND=1. The first two must be identical, and equal to the scores on disk. The third is a
    measurement: how far the Pearson-VST moves when counts are rounded instead of truncated."""
    S = a.scratch / "vst"
    val = Path(subprocess.run(
        [PY, "-c", "import sys; sys.path.insert(0, 'lib'); from gene_utils import load_config, resolve_path;"
                   "c = load_config(); print(resolve_path(c, c['deconvolution']['validation_dir']))"],
        cwd=a.new_tree, env=dict(os.environ, PIPELINE_ROOT=str(a.new_tree)), capture_output=True, text=True,
        check=True).stdout.strip())
    script, out = "deconvolution/R/score_z_vst.R", "z_vst_focal.tsv"
    sweeps = sorted(p for p in val.glob("SWEEP_*") if (p / "sweep_spec.json").exists()
                    and (p / "mixtures/true_z_focal.csv").exists() and (p / "results/pred_z").is_dir())
    row("vst", "sweeps on disk", str(val), "", "", "INFO", f"{len(sweeps)}: " + ", ".join(p.name for p in sweeps))

    def scores(path):
        with open(path) as fh:
            return {r["mixture"]: float(r["pearson_vst"]) for r in csv.DictReader(fh, delimiter="\t")
                    if r["pearson_vst"] not in ("", "NA")}

    for sw in sweeps:
        focal = json.load(open(sw / "sweep_spec.json"))["focal"]
        runs = {}
        for tag, tree, env in (("old", a.old_tree, {}), ("new", a.new_tree, {}),
                               ("new_round", a.new_tree, {"SCORE_Z_VST_ROUND": "1"})):
            stage = S / sw.name / tag
            stage.mkdir(parents=True, exist_ok=True)
            for sub in ("mixtures", "results"):
                if not (stage / sub).exists():
                    os.symlink(sw / sub, stage / sub)
            rc, _ = run(rscript(tree, script) + [stage, focal], tree, S / sw.name / f"{tag}.log", env=env)
            runs[tag] = stage / "scores" / out if rc == 0 else None
        if runs["old"] is None or runs["new"] is None:
            row("vst", f"{sw.name}: run", "-", "", "", "RUN FAILED", f"focal {focal}")
            continue
        compare_file("vst", f"{sw.name}/scores/{out} (default)", runs["old"], runs["new"], "old script (44332e6)")
        if (sw / "scores" / out).exists():
            compare_file("vst", f"{sw.name}/scores/{out} against the scores on disk", sw / "scores" / out,
                         runs["new"], str(val.name))
        if runs["new_round"] is not None:
            t, r = scores(runs["new"]), scores(runs["new_round"])
            common = sorted(set(t) & set(r))
            if common:
                d = [r[m] - t[m] for m in common]
                med = lambda v: sorted(v)[len(v) // 2]                                   # noqa: E731
                row("vst", f"{sw.name}: rounding instead of truncation (switch, off by default)",
                    "truncation (default)", "", "", "INFO",
                    f"focal {focal}; {len(common)} mixtures; median Pearson-VST {med([t[m] for m in common]):.4f} "
                    f"truncated, {med([r[m] for m in common]):.4f} rounded; change per mixture: "
                    f"min {min(d):+.4f}, median {med(d):+.4f}, max {max(d):+.4f}")
        else:
            row("vst", f"{sw.name}: rounding run", "-", "", "", "RUN FAILED", f"focal {focal}")


# ----------------------------------------------------------------------------- purity-sweep generation
def check_sweepgen(a, sweep="SWEEP_lung_at2_holdout"):
    """deconvolution/make_purity_sweep.py, OLD against NEW, with the command pipeline/run_stage14.py builds
    for one sweep (LUNG alveolar type II cells: the one default sweep in which the rounding-drift repair
    leaves a negative background count). The NEW script only adds a printed note; every output file must be
    identical to the OLD script's, and the note must name the mixtures whose true_fractions.tsv holds a
    negative cell fraction. The on-disk sweep is compared too, for information."""
    S = a.scratch / "sweepgen"
    val = Path(subprocess.run(
        [PY, "-c", "import sys; sys.path.insert(0, 'lib'); from gene_utils import load_config, resolve_path;"
                   "c = load_config(); print(resolve_path(c, c['deconvolution']['validation_dir']))"],
        cwd=a.new_tree, env=dict(os.environ, PIPELINE_ROOT=str(a.new_tree)), capture_output=True, text=True,
        check=True).stdout.strip())
    spec = json.load(open(val / sweep / "sweep_spec.json"))
    tissue = sweep.split("_")[1]
    logs = {}
    for tag, tree in (("old", a.old_tree), ("new", a.new_tree)):
        out = S / tag
        cmd = [PY, "deconvolution/make_purity_sweep.py", "--mode", "holdout",
               "--ref-source", a.main_root / spec["reference_dir"], "--tissue", tissue,
               "--focal-type", spec["focal"], "--out", out, "--purity-grid", spec["purity_grid"],
               "--reps", str(spec["reps"]), "--cells-per-mixture", str(spec["cells_per_mixture"])]
        rc, logs[tag] = run(cmd, tree, S / f"{tag}.log")
        if rc != 0:
            row("sweepgen", f"{tag} run", "-", "", "", "RUN FAILED", f"exit {rc}")
            return
    files = sorted(str(p.relative_to(S / "old")) for p in (S / "old").rglob("*") if p.is_file())
    n_same = 0
    for f in files:
        x, y = md5(S / "old" / f), (md5(S / "new" / f) if (S / "new" / f).exists() else "")
        n_same += x == y
        if x != y or f.endswith(("true_fractions.tsv", "pseudobulk_counts.mtx", "true_z_focal.csv")):
            row("sweepgen", f, "old script (44332e6)", x, y, "IDENTICAL" if x == y else "DIFFERENT")
    row("sweepgen", f"{len(files)} files of the sweep (mixtures, true Z, split reference)", "old script (44332e6)",
        "", "", "IDENTICAL" if n_same == len(files) else "DIFFERENT", f"{n_same} of {len(files)} identical")
    for f in ("mixtures/true_fractions.tsv", "mixtures/pseudobulk_counts.mtx", "mixtures/sweep_design.tsv"):
        x, y = md5(val / sweep / f), md5(S / "new" / f)
        row("sweepgen", f"{f} against the sweep on disk", f"{val.name}/{sweep}", x, y,
            "IDENTICAL" if x == y else "INFO", "" if x == y else "differs from the sweep on disk (informative only)")
    with open(S / "new/mixtures/true_fractions.tsv") as fh:
        rd = list(csv.DictReader(fh, delimiter="\t"))
    neg = sorted({r["mixture"] for r in rd for c in r if c.startswith("cellfrac__") and float(r[c]) < 0},
                 key=lambda m: int(m[3:]))
    note = next((ln for ln in logs["new"].splitlines() if ln.startswith("NOTE: ")), "")
    named = all(m in note for m in neg) and f"{len(neg)} of {len(rd)} mixtures" in note
    row("sweepgen", "printed note names the mixtures with a negative cell fraction", "true_fractions.tsv", "", "",
        "IDENTICAL" if (neg and named) or (not neg and not note) else "DIFFERENT",
        f"negative in {neg or 'none'}; note: {note[:160] or 'none'}")
    row("sweepgen", "old script prints no such note", "-", "", "",
        "IDENTICAL" if "NOTE: " not in logs["old"] else "DIFFERENT", "")
    for tag in ("old", "new"):                                   # the split references are large
        for big in (S / tag).rglob("reference_counts.mtx"):
            big.unlink()


CHECKS = {"refs": check_refs, "stage3": check_stage3, "rin": check_rin, "pheno": check_pheno,
          "augur": check_augur, "atlas": check_atlas, "vst": check_vst, "sweepgen": check_sweepgen}
GOOD = {"IDENTICAL", "WRITTEN", "STOPPED", "INFO", "COUNT", "KEYS", "PATH ONLY", "EXPECTED DIFFERENCE"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old-tree", required=True, type=Path)
    ap.add_argument("--new-tree", required=True, type=Path)
    ap.add_argument("--scratch", required=True, type=Path)
    ap.add_argument("--record-dir", required=True, type=Path)
    ap.add_argument("--old-commit", default="44332e6")
    ap.add_argument("--new-commit", default="")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=list(CHECKS))
    a = ap.parse_args()
    a.old_tree, a.new_tree, a.scratch = a.old_tree.resolve(), a.new_tree.resolve(), a.scratch.resolve()
    a.data = (a.new_tree / "data").resolve()
    a.main_root = a.data.parent                     # the repository whose data/ the trees link to
    a.record_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in a.only:
        print(f"\n=== {name} ===", flush=True)
        n0, t0 = len(ROWS), time.time()
        try:
            CHECKS[name](a)
        except Exception as e:                      # one failed check must not hide the others
            row(name, "check raised", "-", "", "", "ERROR", f"{type(e).__name__}: {e}")
        mine = ROWS[n0:]
        bad = [r for r in mine if r["result"] not in GOOD]
        summary.append({"check": name, "rows": len(mine), "not_as_expected": len(bad),
                        "old_commit": a.old_commit, "new_commit": a.new_commit,
                        "seconds": round(time.time() - t0), "job": os.environ.get("SLURM_JOB_ID", ""),
                        "what_differs": "; ".join(f"{r['item']}: {r['result']}" for r in bad)})
    suffix = "" if set(a.only) == set(CHECKS) else "_" + "_".join(a.only)
    for fname, rows in ((f"neutrality_checks{suffix}.tsv", ROWS), (f"neutrality_checks{suffix}_summary.tsv", summary)):
        with open(a.record_dir / fname, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
            w.writeheader(); w.writerows(rows)
    print("\n=== summary ===")
    for s in summary:
        print(f"  {s['check']:<8} rows {s['rows']:>3}  not as expected {s['not_as_expected']}  {s['what_differs']}")


if __name__ == "__main__":
    main()
