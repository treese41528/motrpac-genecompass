#!/usr/bin/env python3
"""build_references_from_config.py -- build every BayesPrism reference from tissue_references.yaml.

Replaces the ad-hoc build_all_references.sh / build_references_v2.sh hardcoded job lists with a single
config-driven builder. For each buildable tissue it constructs the exact build_reference.py invocation
from the config's filters (organism / conditions / title / sample_ids / dedup / label-scheme / gene-join),
builds into reference_dir, then runs the post-build selection check (cells_meta used exactly expect.sample_ids).
Shared references (SKMGN+SKMVL -> one Ma muscle build) are built once. Pooled lung uses build_lung_pooled.py.

Loads many h5ads -> RUN ON A COMPUTE NODE (see --submit).

  python deconvolution/build_references_from_config.py                 # dry-run: print the plan
  python deconvolution/build_references_from_config.py --run           # build locally (on a compute node)
  python deconvolution/build_references_from_config.py --run --tissue CORTEX HEART --force
  python deconvolution/build_references_from_config.py --submit        # sbatch one job that builds all
"""
import argparse, os, shlex, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deconvolution"))
os.environ.setdefault("PIPELINE_ROOT", str(ROOT))
import validate_selection as V  # noqa: E402

PY = os.environ.get("DECONV_PYTHON") or sys.executable
BUILDABLE = {"ready", "needs-build", "proposed"}


def build_cmd(cfg):
    """The exact command that builds this tissue's reference into cfg['reference_dir']."""
    out = cfg["reference_dir"]
    if cfg.get("pooled"):
        return [PY, "deconvolution/build_lung_pooled.py", "--out", out]
    c = [PY, "deconvolution/build_reference.py",
         "--study", cfg["study"], "--tissue", cfg["tissue"],
         "--organism", cfg.get("organism", "Rattus norvegicus"),
         "--label-scheme", cfg.get("label_scheme", "none"),
         "--gene-join", cfg.get("gene_join", "inner"),
         "--min-gene-cells", str(cfg.get("min_gene_cells", 0)),
         "--out", out]
    if cfg.get("conditions"):
        c += ["--conditions", ",".join(cfg["conditions"])]
    # use --opt=value so a regex starting with '-' (e.g. the Ma '-Y' arm) is not mistaken for a flag
    if cfg.get("title_include"):
        c += [f"--title-include={cfg['title_include']}"]
    if cfg.get("title_exclude"):
        c += [f"--title-exclude={cfg['title_exclude']}"]
    if cfg.get("sample_ids"):
        c += ["--sample-ids", ",".join(cfg["sample_ids"])]
    if not cfg.get("dedup_gsm", True):
        c += ["--no-dedup-gsm"]
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="execute the builds locally (on a compute node)")
    ap.add_argument("--submit", action="store_true", help="sbatch ONE job that builds all references")
    ap.add_argument("--tissue", nargs="*", help="subset of BULK codes (default: all buildable)")
    ap.add_argument("--force", action="store_true", help="rebuild even if the reference already exists")
    ap.add_argument("--account", default="reese18")
    ap.add_argument("--partition", default="a100-40gb")
    ap.add_argument("--mem", default="128G")
    ap.add_argument("--cores", type=int, default=16)
    ap.add_argument("--time", default="6:00:00")
    args = ap.parse_args()

    cfg_all = V.load_config()
    inv = V._inventory()
    codes = args.tissue or list(cfg_all)

    # dedupe by reference_dir so a shared build (muscle) runs once
    plan, seen_dirs = [], {}
    for code in codes:
        cfg = cfg_all.get(code)
        if not cfg or cfg.get("status") not in BUILDABLE or not cfg.get("reference_dir"):
            continue
        if cfg.get("data_present") is False:
            print(f"[skip] {code}: needs ingestion first ({cfg.get('study')}) -- not buildable from corpus")
            continue
        rd = cfg["reference_dir"]
        if rd in seen_dirs:
            print(f"[shared] {code}: reuses {seen_dirs[rd]}'s build ({rd})")
            continue
        seen_dirs[rd] = code
        plan.append((code, cfg))

    if args.submit:
        sub = " ".join(shlex.quote(t) for t in (args.tissue or []))
        inner = (f"set -euo pipefail; cd {ROOT}; export PIPELINE_ROOT={ROOT}; "
                 f'eval "$(python3 deconvolution/_config_sh.py 2>/dev/null || true)"; '
                 f"{shlex.quote(PY)} deconvolution/build_references_from_config.py --run "
                 f"{'--force ' if args.force else ''}{('--tissue '+sub) if sub else ''}")
        sb = ["sbatch", f"--account={args.account}", f"--partition={args.partition}", "--nodes=1",
              "--ntasks=1", f"--cpus-per-task={args.cores}", f"--mem={args.mem}", "--gres=gpu:1",
              f"--time={args.time}", "--job-name=build_refs",
              "--output=logs/build_refs_%j.out", "--error=logs/build_refs_%j.err", f"--wrap={inner}"]
        (ROOT / "logs").mkdir(exist_ok=True)
        r = subprocess.run(sb, cwd=ROOT, capture_output=True, text=True)
        print((r.stdout or r.stderr).strip())
        return

    print(f"\n=== build plan: {len(plan)} reference(s) ({'RUN' if args.run else 'DRY-RUN'}) ===")
    fails = []
    for code, cfg in plan:
        rd = cfg["reference_dir"]
        built = (ROOT / rd / "cells_meta.tsv").exists()
        cmd = build_cmd(cfg)
        print(f"\n[{code}] -> {rd}" + ("  (exists)" if built else ""))
        print("   " + " ".join(shlex.quote(x) for x in cmd))
        if not args.run:
            continue
        if built and not args.force:
            print("   [skip] already built (use --force to rebuild)")
        else:
            env = dict(os.environ, PIPELINE_ROOT=str(ROOT))
            if cfg.get("allow_nonnative"):
                env["ALLOW_NONNATIVE_REF"] = "1"
            rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
            if rc != 0:
                fails.append(f"{code}: build exited {rc}"); print(f"   BUILD FAILED ({rc})"); continue
        # post-build: assert the built reference used exactly expect.sample_ids
        v, lines = V.check_tissue(code, cfg, inv, built_only=True)
        print(f"   post-build validate: {v}")
        for ln in lines:
            print(f"      {ln}")
        if v == "FAIL":
            fails.append(f"{code}: post-build validation FAILED")

    if args.run:
        print(f"\n=== {len(plan)-len(fails)}/{len(plan)} built+validated OK ===")
        if fails:
            print("FAILURES:")
            for f in fails:
                print("  " + f)
        sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
