#!/usr/bin/env python3
"""assemble_arm.py -- how the refs4 / refs4all results trees were ASSEMBLED from earlier arms by links (the record).

The assembled 16-tissue tree (results_refs4all/, genecompass_input_v2_refs4all/ (+ _human/)) was not produced by
one run: tissues whose reference did not change were LINKED in from the arm that had already deconvolved and
embedded them, and only the cross-tissue steps (gate, pooled IHW/repfdr DE, transfer comparison, GRNs, Stage 14)
were rerun over the assembled set. This script is the committed form of the two local setup scripts that did it
(slurm/analysis/refs4_setup.sh, refs4all_setup.sh), with every path taken from the config profiles:

  refs4     (REFERENCE_RELABEL_PREREG.md §8; config/profiles/refs4_14.yaml)
            BAT, HEART, HYPOTH, TESTES linked from the July arm (config/profiles/july_refs.yaml): their references
            did not change (REFS4_RERUN_RESULTS.md). The other 10 tissues were run in place.
  refs4all  (§16; config/profiles/refs_v4all.yaml; the default config since 2026-11-03 is its clean rebuild,
            REFS4ALL_REBUILD_RESULTS.md)
            SPLEEN and OVARY, first run inside the refs4 folders, MOVED into refs4all; the other 14 tissues linked
            from refs4 (link targets resolved, so BAT/HEART/HYPOTH/TESTES point at the July files directly).

Layout: results_dir/motrpac/<TISSUE> is a directory symlink (read-only Stage 10 inputs: pred_z, fractions);
genecompass_input_dir/<label> and its _human twin are REAL directories whose files are symlinks, so anything
written later (gate, DE, embeddings of a rerun) lands in the new arm, never in the source arm. A listing of
every linked target (path, size, mtime) is written so a later run can be checked for writes through the links
(--verify compares against it).

The CLEAN path does not need this: pipeline/run_deconv_all.py builds all 16 tissues straight into the config's
folders (DECONV_REBUILD_RUNBOOK.md §0). Use this only to reproduce or audit the assembled tree.

  python deconvolution/assemble_arm.py refs4all            # dry run: print what would be linked/moved
  python deconvolution/assemble_arm.py refs4all --run      # do it (refuses if the target folders exist)
  python deconvolution/assemble_arm.py refs4all --verify   # compare linked targets with the saved listing
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(os.environ.get("PIPELINE_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "lib"))
from gene_utils import load_config, resolve_path  # noqa: E402

PROFILES = {"july": "config/profiles/july_refs.yaml",
            "refs4": "config/profiles/refs4_14.yaml",
            "refs4all": "config/profiles/refs_v4all.yaml"}

# target arm -> (link: {source arm: [tissues]}, move: {source arm: [tissues]})
RECIPES = {
    "refs4": {"link": {"july": ["BAT", "HEART", "HYPOTH", "TESTES"]}, "move": {}},
    "refs4all": {"link": {"refs4": ["BAT", "BLOOD", "CORTEX", "HEART", "HIPPOC", "HYPOTH", "KIDNEY", "LIVER",
                                    "LUNG", "SKMGN", "SKMVL", "SMLINT", "TESTES", "WATSC"]},
                 "move": {"refs4": ["SPLEEN", "OVARY"]}},
}


def arm_dirs(arm):
    """(results motrpac dir, genecompass_input dir, its _human twin) of an arm, from its profile."""
    old = os.environ.get("PIPELINE_PROFILE")
    os.environ["PIPELINE_PROFILE"] = PROFILES[arm]
    try:
        c = load_config()
    finally:
        if old is None:
            os.environ.pop("PIPELINE_PROFILE", None)
        else:
            os.environ["PIPELINE_PROFILE"] = old
    d = c["deconvolution"]
    gc = resolve_path(c, d["genecompass_input_dir"])
    return resolve_path(c, d["results_dir"]) / "motrpac", gc, Path(str(gc) + "_human")


def listing_path(arm):
    return ROOT / "tmp" / f"{arm}_linked_targets_before.tsv"


def target_listing(arm):
    """path, size, mtime of every file reachable under the source arms of this recipe (follows links)."""
    rows = []
    srcs = set(RECIPES[arm]["link"]) | {"july"}
    for s in sorted(srcs):
        for d in arm_dirs(s):
            if d.exists():
                # follow directory links, as the setup scripts' `find -L` did: refs4's BAT/HEART/HYPOTH/TESTES
                # folders are links into the July arm, and their files are listed under the refs4 path too
                for dirpath, _, names in os.walk(d, followlinks=True):
                    for name in sorted(names):
                        p = Path(dirpath) / name
                        try:
                            if p.is_file():
                                st = p.stat()
                                rows.append(f"{p}\t{st.st_size}\t{st.st_mtime_ns}")
                        except OSError:
                            pass
    return rows


def parse_listing(rows):
    """{path relative to ROOT: (size, mtime in ns)}.

    Reads both forms: the shell setup scripts wrote absolute paths and `stat -c %.Y` seconds with ten decimals;
    target_listing() writes st_mtime_ns. mtimes are compared in whole microseconds, the precision both keep."""
    out = {}
    for r in rows:
        if not r.strip():
            continue
        path, size, mtime = r.split("\t")
        p = Path(path)
        rel = str(p.relative_to(ROOT)) if p.is_absolute() and p.is_relative_to(ROOT) else path
        if "." in mtime:                             # decimal seconds, parsed as a string: a float drops the µs
            sec, frac = mtime.split(".")
            us = int(sec) * 1_000_000 + int((frac + "000000")[:6])
        else:                                        # st_mtime_ns
            us = int(mtime) // 1000
        out[rel] = (int(size), us)
    return out


def moved_sources(arm):
    """Source folders of the tissues this recipe MOVES: they leave the source arm, so they drop out of the listing."""
    out = []
    for src, tissues in RECIPES[arm]["move"].items():
        sres, sgc, shu = arm_dirs(src)
        for t in tissues:
            out += [(d / n).relative_to(ROOT) for d, n in ((sres, t), (sgc, t.lower()), (shu, t.lower()))]
    return out


def plan(arm):
    res, gc, hu = arm_dirs(arm)
    ops = []
    for src, tissues in RECIPES[arm]["move"].items():
        sres, sgc, _ = arm_dirs(src)
        for t in tissues:
            ops.append(("move", sres / t, res / t))
            ops.append(("move", sgc / t.lower(), gc / t.lower()))
    for src, tissues in RECIPES[arm]["link"].items():
        sres, sgc, shu = arm_dirs(src)
        for t in tissues:
            ops.append(("link_dir", sres / t, res / t))
            ops.append(("link_files", sgc / t.lower(), gc / t.lower()))
            ops.append(("link_files", shu / t.lower(), hu / t.lower()))
    return (res, gc, hu), ops


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm", choices=sorted(RECIPES))
    ap.add_argument("--run", action="store_true", help="create the links / moves (default: dry run)")
    ap.add_argument("--verify", action="store_true", help="check the linked targets against the saved listing")
    a = ap.parse_args()

    if a.verify:
        before = listing_path(a.arm)
        if not before.exists():
            sys.exit(f"no saved listing: {before}")
        old, new = parse_listing(before.read_text().splitlines()), parse_listing(target_listing(a.arm))
        moved = tuple(str(p) + "/" for p in moved_sources(a.arm))
        gone = sorted(p for p in set(old) - set(new) if p.startswith(moved))
        changed = sorted(p for p in set(old) | set(new)
                         if not p.startswith(moved) and old.get(p) != new.get(p))
        print(f"{len(new)} linked-target files checked" + (f"; {len(gone)} files of the moved tissues have left "
              "their source arm (expected)" if gone else ""))
        print("none changed (listing identical)" if not changed else f"WARNING: {len(changed)} linked-target files changed:")
        for p in changed[:50]:
            print(f"  {p}  before {old.get(p, 'absent')}  now {new.get(p, 'absent')}")
        sys.exit(1 if changed else 0)

    (res, gc, hu), ops = plan(a.arm)
    targets = [res.parent, gc, hu]
    if a.run:
        clash = [str(p) for p in targets if p.exists()]
        if clash:
            sys.exit(f"FATAL: target folder(s) exist: {clash}")
        for p in (res, gc, hu):
            p.mkdir(parents=True)
    for kind, src, dst in ops:
        print(f"{kind:10s} {src} -> {dst}")
        if not a.run:
            continue
        if kind == "move":
            shutil.move(str(src), str(dst))
        elif kind == "link_dir":
            dst.symlink_to(src.resolve())
        else:
            dst.mkdir()
            for f in sorted(src.iterdir()):
                (dst / f.name).symlink_to(f.resolve())
    if a.run:
        lp = listing_path(a.arm)
        lp.parent.mkdir(exist_ok=True)
        rows = target_listing(a.arm)
        lp.write_text("\n".join(rows) + "\n")
        print(f"{len(rows)} linked-target files listed -> {lp}")
    else:
        print("(dry run; pass --run to apply)")


if __name__ == "__main__":
    main()
