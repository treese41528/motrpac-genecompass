#!/usr/bin/env python3
"""run_deconv_all.py — from-scratch-reproducible Stage-8 + Stage-9 driver.

Reads `deconvolution/tissue_references.yaml` (the CANONICAL tissue->reference map) and runs the
deconvolution -> GeneCompass-embedding chain for every MoTrPAC tissue with the CORRECT reference, so a
fresh run cannot silently use the wrong study (the bug that cost us liver Visium / engineered lung).
Before each tissue it re-validates the reference with `reference_qc.py --fail` (native/adult/healthy/
single-cell), and refuses to run a missing or contaminated reference — pointing at the manifest's build
command. This is the single committed entry point that ties the manifest to the pipeline.

  python pipeline/run_deconv_all.py --dry-run                     # validate refs + print the plan
  python pipeline/run_deconv_all.py --submit                      # sbatch one SLURM job per tissue
  python pipeline/run_deconv_all.py --submit --tissue LIVER LUNG  # a subset
  python pipeline/run_deconv_all.py --submit --from-step 2        # re-run (reuse lifted bulk)

Requires PyYAML (in the project venv). Run with the venv python.
"""
import argparse, shlex, subprocess, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "deconvolution" / "tissue_references.yaml"


def load_manifest():
    d = yaml.safe_load(open(MANIFEST))
    return {k: v for k, v in d.items() if isinstance(v, dict) and "reference_dir" in v}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submit", action="store_true", help="sbatch one job per tissue")
    ap.add_argument("--dry-run", action="store_true", help="validate refs + print the plan; run nothing")
    ap.add_argument("--tissue", nargs="*", help="subset of BULK codes (default: all in the manifest)")
    ap.add_argument("--from-step", type=int, default=1, choices=[1, 2],
                    help="run_stage8 --from: 1 = full incl bulk liftover (from scratch); 2 = reuse lifted bulk (re-run)")
    ap.add_argument("--account", default="reese18")
    ap.add_argument("--partition", default="a100-40gb")
    ap.add_argument("--mem", default="128G")
    ap.add_argument("--cores", type=int, default=16)
    ap.add_argument("--time", default="8:00:00")
    ap.add_argument("--skip-qc", action="store_true", help="skip the reference_qc gate (NOT recommended)")
    args = ap.parse_args()
    if not (args.submit or args.dry_run):
        ap.error("give --submit or --dry-run")
    (ROOT / "logs").mkdir(exist_ok=True)

    man = load_manifest()
    tissues = args.tissue or list(man)
    py = sys.executable
    problems, planned = [], []

    for t in tissues:
        if t not in man:
            problems.append(f"{t}: not in the manifest ({MANIFEST.name})"); continue
        ref = man[t]["reference_dir"]
        if not (ROOT / ref).is_dir():
            problems.append(f"{t}: reference MISSING: {ref}\n       build it: {man[t].get('build', '(see manifest)')}")
            continue
        if not args.skip_qc:
            qc = subprocess.run([py, str(ROOT / "deconvolution" / "reference_qc.py"),
                                 "--ref-dir", ref, "--fail"], cwd=ROOT)
            if qc.returncode != 0:
                problems.append(f"{t}: reference FAILED reference_qc (non-native): {ref}"); continue
        label = t.lower()
        inner = (f"{py} pipeline/run_stage8.py --tissue {t} --ref-dir {shlex.quote(ref)} "
                 f"--label {label} --from {args.from_step} --n-cores {args.cores} && "
                 f"{py} pipeline/run_stage9.py --label {label} --device cuda")
        planned.append((t, ref, label, inner))

    if problems:
        print("\n=== PROBLEMS — fix before running (from tissue_references.yaml) ===")
        for p in problems:
            print("  " + p)
        # still show the ok plan, but exit non-zero
    print(f"\n=== plan: {len(planned)}/{len(tissues)} tissues OK "
          f"(from-step={args.from_step}, {'SUBMIT' if args.submit else 'DRY-RUN'}) ===")
    for t, ref, label, inner in planned:
        print(f"[{t:8s}] ref={ref}")
        if args.dry_run:
            print(f"           {inner}")
            continue
        wrap = (f"set -euo pipefail; cd {ROOT}; export PROJECT_ROOT={ROOT} PIPELINE_ROOT={ROOT}; "
                f"eval \"$(python3 deconvolution/_config_sh.py 2>/dev/null || true)\"; {inner}")
        sb = ["sbatch", f"--account={args.account}", f"--partition={args.partition}",
              "--nodes=1", "--ntasks=1", f"--cpus-per-task={args.cores}", f"--mem={args.mem}",
              "--gres=gpu:1", "--gpus-per-task=1", "--qos=normal", f"--time={args.time}",
              f"--job-name=deconv_{label}",
              f"--output=logs/deconv_{label}_%j.out", f"--error=logs/deconv_{label}_%j.err",
              f"--wrap={wrap}"]
        r = subprocess.run(sb, cwd=ROOT, capture_output=True, text=True)
        print(f"           {(r.stdout or r.stderr).strip()}")
    if planned and args.submit:
        print("\nAfter all tissues finish, run the cross-tissue layer: redetect_redE.slurm (Stage 11 + "
              "Augur + hotspots + Stage 10) then run_stage12.slurm (transfer).")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
