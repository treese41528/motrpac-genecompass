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

Outputs go STRAIGHT into the active config's folders -- deconvolution.results_dir/motrpac/<TISSUE> and
deconvolution.genecompass_input_dir/<label> (default config: the 16-tissue _refs4all_rebuild folders) -- with no links.
A tissue whose output folder is a symlink (an arm assembled by deconvolution/assemble_arm.py) is refused,
because writing would go through the link into another arm; pass --allow-links to override.

The corpus annotation inventory (reports/annotations/annotation_inventory.tsv, gitignored) is OPTIONAL: when
it is absent, gate 1 checks only the built reference's samples against `expect:` and gate 2 (reference_qc,
which reads GEO titles from the inventory) is skipped with a note. It is needed only to (re)build corpus
references from the single-cell corpus.

  python pipeline/run_deconv_all.py --dry-run                 # validate + print the plan; run nothing
  python pipeline/run_deconv_all.py --build --submit          # build any missing refs, then deconvolve
  python pipeline/run_deconv_all.py --submit --tissue LIVER LUNG
  python pipeline/run_deconv_all.py --validate-only           # just run the selection guard, exit
  python pipeline/run_deconv_all.py --run-local --tissue OVARY  # run serially here (on a GPU compute node)

SLURM account/partition/QOS come from deconvolution/setup/site.env (MGC_SLURM_*; see slurm_site.py) unless
--account/--partition/--qos are given. Requires PyYAML (project venv). Run with the venv python.
"""
import argparse, os, shlex, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PIPELINE_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "deconvolution"))
sys.path.insert(0, str(ROOT / "lib"))
import validate_selection as V  # noqa: E402
from slurm_site import sbatch_site_args  # noqa: E402
from gene_utils import load_config, resolve_path  # noqa: E402

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
    ap.add_argument("--run-local", action="store_true",
                    help="run each tissue's Stage 8 + 9 serially in this process (use inside a GPU compute job)")
    ap.add_argument("--allow-links", action="store_true",
                    help="write even if a tissue's output folder is a symlink into another arm (NOT recommended)")
    ap.add_argument("--account", default=None, help="sbatch --account (default: MGC_SLURM_ACCOUNT in site.env)")
    ap.add_argument("--partition", default=None, help="sbatch --partition (default: MGC_SLURM_PARTITION)")
    ap.add_argument("--qos", default=None, help="sbatch --qos (default: MGC_SLURM_QOS)")
    ap.add_argument("--mem", default="128G")
    ap.add_argument("--cores", type=int, default=16)
    ap.add_argument("--time", default="8:00:00")
    ap.add_argument("--skip-qc", action="store_true", help="skip the reference_qc gate (NOT recommended)")
    args = ap.parse_args()
    if not (args.submit or args.dry_run or args.validate_only or args.run_local):
        ap.error("give --submit, --run-local, --dry-run, or --validate-only")
    (ROOT / "logs").mkdir(exist_ok=True)

    cfg_all = V.load_config()
    inv = V._inventory()
    have_inv = V.inventory_available()
    pcfg = load_config()                       # pipeline config (PIPELINE_PROFILE honoured)
    pd_ = pcfg["deconvolution"]
    res_root = resolve_path(pcfg, pd_["results_dir"]) / "motrpac"
    gc_root = resolve_path(pcfg, pd_["genecompass_input_dir"])
    print(f"results -> {res_root}\npseudo-cells/embeddings -> {gc_root}")
    if not have_inv:
        print("[note] no annotation inventory: gate 1 = post-build sample check; gate 2 (reference_qc) skipped")
    py = sys.executable
    tissues = args.tissue or list(cfg_all)
    runnable = set(RUNNABLE) | ({"proposed"} if args.include_proposed else set())

    # optional: build any missing references first (config-driven; runs the same validated builder)
    if args.build and not args.dry_run:
        to_build = [t for t in tissues if cfg_all.get(t, {}).get("status") in runnable
                    and not (ROOT / (cfg_all[t].get("reference_dir") or "x") / "cells_meta.tsv").exists()]
        if to_build:
            print(f"=== building {len(to_build)} missing reference(s): {to_build} ===")
            rc = subprocess.run([py, "deconvolution/build_references_from_config.py", "--run",
                                 "--tissue", *to_build], cwd=ROOT).returncode
            if rc:
                print(f"[warn] reference build exited {rc}; tissues whose reference is missing are reported below")

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
        # never write through a link into another arm (assembled trees; see deconvolution/assemble_arm.py)
        label = t.lower()
        linked = [str(p) for p in (res_root / t, gc_root / label) if p.is_symlink()]
        linked += [str(p) for d in (res_root / t, gc_root / label) if d.is_dir() and not d.is_symlink()
                   for p in d.iterdir() if p.is_symlink()]
        if linked and not args.allow_links:
            problems.append(f"{t}: output folder is or holds symlinks (assembled arm): {linked[:3]}\n"
                            f"       remove the link (or use a fresh results_dir), or pass --allow-links")
            continue
        # GATE 2: reference_qc (native/adult/healthy) unless intrinsic-sort override
        if not args.skip_qc and not cfg.get("allow_nonnative") and have_inv:
            qc = subprocess.run([py, str(ROOT / "deconvolution" / "reference_qc.py"),
                                 "--ref-dir", ref, "--fail"], cwd=ROOT)
            if qc.returncode != 0:
                problems.append(f"{t}: reference_qc FAILED (non-native): {ref}"); continue
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

    # the job inherits the PIPELINE_PROFILE of this shell, so every tissue writes to the same arm
    prof = os.environ.get("PIPELINE_PROFILE", "")
    envs = (f"export PROJECT_ROOT={shlex.quote(str(ROOT))} PIPELINE_ROOT={shlex.quote(str(ROOT))} "
            f"DECONV_PYTHON={shlex.quote(py)} TMPDIR={shlex.quote(str(ROOT / 'tmp'))}"
            + (f" PIPELINE_PROFILE={shlex.quote(prof)}" if prof else "") + "; ")
    failed = []
    for t, ref, label, inner in planned:
        if args.run_local:
            print(f"\n=== [{t}] Stage 8 + 9 (local) ===", flush=True)
            env = dict(os.environ, PROJECT_ROOT=str(ROOT), PIPELINE_ROOT=str(ROOT), DECONV_PYTHON=py)
            if subprocess.run(["bash", "-c", f"set -euo pipefail; {inner}"], cwd=ROOT, env=env).returncode:
                failed.append(t)
            continue
        wrap = f"set -euo pipefail; cd {shlex.quote(str(ROOT))}; mkdir -p tmp; {envs}{inner}"
        sb = (["sbatch"] + sbatch_site_args(args.account, args.partition, args.qos, gpu=True)
              + ["--nodes=1", "--ntasks=1", f"--cpus-per-task={args.cores}", f"--mem={args.mem}",
                 f"--time={args.time}", f"--job-name=deconv_{label}",
                 f"--output=logs/deconv_{label}_%j.out", f"--error=logs/deconv_{label}_%j.err",
                 f"--wrap={wrap}"])
        r = subprocess.run(sb, cwd=ROOT, capture_output=True, text=True)
        print(f"[{t:8s}] {(r.stdout or r.stderr).strip()}")
    if failed:
        print(f"\nFAILED (local): {failed}")
        problems.append(f"local run failed: {failed}")
    if planned and (args.submit or args.run_local):
        print("\nAfter all tissues finish, run the cross-tissue layer in order (deconvolution/DECONV_REBUILD_RUNBOOK.md "
              "§0): rat gate (subspace_probe + augur_prep + run_augur.R + corroborate_summary) -> run_stage10 -> "
              "run_stage12 -> run_stage13 -> run_stage14. Reference changes perturb the pooled IHW/repfdr fit across "
              "all DE blocks -- re-run Stage 10-14 together.")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
