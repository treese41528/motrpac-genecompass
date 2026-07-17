#!/usr/bin/env python3
"""run_deconv_all.py -- from-scratch-reproducible Stage-8 + Stage-9 driver.

Reads `deconvolution/tissue_references.yaml` (the CANONICAL tissue->reference map, schema v3) and runs the
deconvolution -> GeneCompass-embedding chain for every MoTrPAC tissue with the CORRECT reference, so a fresh
run cannot silently use the wrong study or wrong samples (the bug that cost us liver Visium / engineered lung /
cross-species cortex). Before each tissue it runs TWO gates:
  1. validate_selection.check_tissue  -- asserts the reference selects EXACTLY the samples we specify
     (organism/arm/sample_ids) and that reference_qc dropped only the intended samples; refuses to deconvolve
     a reference that does not match its `expect:` contract. (This is the "validate before deconvolving" guard.)
  2. reference_qc.py --fail            -- native/adult/healthy/single-cell (skipped when allow_nonnative).
Tissues are skipped (not run) when blocked / needs-online-data / (proposed, unless --include-proposed).

  python pipeline/run_deconv_all.py --dry-run                 # validate + print the plan; run nothing
  python pipeline/run_deconv_all.py --build --submit          # build any missing refs, then deconvolve
  python pipeline/run_deconv_all.py --submit --tissue LIVER LUNG
  python pipeline/run_deconv_all.py --validate-only           # just run the selection guard, exit

Requires PyYAML (project venv). Run with the venv python.
"""
import argparse, shlex, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deconvolution"))
import validate_selection as V  # noqa: E402

RUNNABLE = {"ready", "needs-build"}   # + "proposed" with --include-proposed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submit", action="store_true", help="sbatch one job per tissue")
    ap.add_argument("--dry-run", action="store_true", help="validate + print the plan; run nothing")
    ap.add_argument("--validate-only", action="store_true", help="run only the selection guard, then exit")
    ap.add_argument("--build", action="store_true", help="build any missing references first (build_references_from_config)")
    ap.add_argument("--include-proposed", action="store_true", help="also run 'proposed' tissues (e.g. HIPPOC, pending sign-off)")
    ap.add_argument("--tissue", nargs="*", help="subset of BULK codes (default: all runnable)")
    ap.add_argument("--from-step", type=int, default=1, choices=[1, 2])
    ap.add_argument("--account", default="reese18")
    ap.add_argument("--partition", default="a100-40gb")
    ap.add_argument("--mem", default="128G")
    ap.add_argument("--cores", type=int, default=16)
    ap.add_argument("--time", default="8:00:00")
    ap.add_argument("--skip-qc", action="store_true", help="skip the reference_qc gate (NOT recommended)")
    args = ap.parse_args()
    if not (args.submit or args.dry_run or args.validate_only):
        ap.error("give --submit, --dry-run, or --validate-only")
    (ROOT / "logs").mkdir(exist_ok=True)

    cfg_all = V.load_config()
    inv = V._inventory()
    py = sys.executable
    tissues = args.tissue or list(cfg_all)
    runnable = set(RUNNABLE) | ({"proposed"} if args.include_proposed else set())

    # optional: build any missing references first (config-driven; runs the same validated builder)
    if args.build and not args.dry_run:
        to_build = [t for t in tissues if cfg_all.get(t, {}).get("status") in runnable
                    and not (ROOT / (cfg_all[t].get("reference_dir") or "x") / "cells_meta.tsv").exists()]
        if to_build:
            print(f"=== building {len(to_build)} missing reference(s): {to_build} ===")
            subprocess.run([py, "deconvolution/build_references_from_config.py", "--run",
                            "--tissue", *to_build], cwd=ROOT)

    problems, planned, skipped = [], [], []
    for t in tissues:
        cfg = cfg_all.get(t)
        if cfg is None:
            problems.append(f"{t}: not in tissue_references.yaml"); continue
        status = cfg.get("status", "")
        if status not in runnable:
            skipped.append(f"{t}: status={status} (skipped)"); continue
        ref = cfg.get("reference_dir")
        if not ref or not (ROOT / ref / "cells_meta.tsv").exists():
            problems.append(f"{t}: reference NOT built: {ref}\n       build it: "
                            f"python deconvolution/build_references_from_config.py --run --tissue {t}")
            continue
        # GATE 1: selection validation (exact samples + QC-drops-only-intended + organism)
        verdict, lines = V.check_tissue(t, cfg, inv, built_only=False)
        if verdict == "FAIL":
            problems.append(f"{t}: SELECTION VALIDATION FAILED:\n       " + "\n       ".join(lines)); continue
        if verdict == "WARN":
            print(f"  [warn] {t}: " + "; ".join(lines))
        # GATE 2: reference_qc (native/adult/healthy) unless intrinsic-sort override
        if not args.skip_qc and not cfg.get("allow_nonnative"):
            qc = subprocess.run([py, str(ROOT / "deconvolution" / "reference_qc.py"),
                                 "--ref-dir", ref, "--fail"], cwd=ROOT)
            if qc.returncode != 0:
                problems.append(f"{t}: reference_qc FAILED (non-native): {ref}"); continue
        label = t.lower()
        inner = (f"{py} pipeline/run_stage8.py --tissue {t} --ref-dir {shlex.quote(ref)} "
                 f"--label {label} --from {args.from_step} --n-cores {args.cores} && "
                 f"{py} pipeline/run_stage9.py --label {label} --device cuda")
        planned.append((t, ref, label, inner))

    if skipped:
        print("\n=== skipped (not runnable this pass) ===")
        for s in skipped:
            print("  " + s)
    if problems:
        print("\n=== PROBLEMS -- fix before running ===")
        for p in problems:
            print("  " + p)
    print(f"\n=== plan: {len(planned)}/{len(tissues)} tissues VALIDATED + OK "
          f"({'SUBMIT' if args.submit else 'DRY-RUN/VALIDATE'}) ===")
    for t, ref, label, inner in planned:
        print(f"[{t:8s}] ref={ref}")

    if args.validate_only or args.dry_run:
        for t, ref, label, inner in planned:
            if args.dry_run:
                print(f"   {inner}")
        sys.exit(1 if problems else 0)

    for t, ref, label, inner in planned:
        wrap = (f"set -euo pipefail; cd {ROOT}; export PROJECT_ROOT={ROOT} PIPELINE_ROOT={ROOT}; "
                f"eval \"$(python3 deconvolution/_config_sh.py 2>/dev/null || true)\"; {inner}")
        sb = ["sbatch", f"--account={args.account}", f"--partition={args.partition}",
              "--nodes=1", "--ntasks=1", f"--cpus-per-task={args.cores}", f"--mem={args.mem}",
              "--gres=gpu:1", "--gpus-per-task=1", "--qos=normal", f"--time={args.time}",
              f"--job-name=deconv_{label}",
              f"--output=logs/deconv_{label}_%j.out", f"--error=logs/deconv_{label}_%j.err",
              f"--wrap={wrap}"]
        r = subprocess.run(sb, cwd=ROOT, capture_output=True, text=True)
        print(f"[{t:8s}] {(r.stdout or r.stderr).strip()}")
    if planned and args.submit:
        print("\nAfter all tissues finish, run the cross-tissue layer (Stage 10/11 + Augur + hotspots), "
              "then run_stage12 (transfer). Reference changes perturb the pooled IHW/repfdr fit across all "
              "185 DE blocks -- re-run Stage 10-12 together.")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
