#!/usr/bin/env python3
"""
run_stage6.py — Stage 6 Orchestrator: Prior Knowledge Embeddings

Runs two steps in sequence:
  Step 1: build_coexp_embedding.py   (slow — scans 9.5M cells, ~2–6 hours)
  Step 2: build_family_embedding.py  (fast — HGNC + gene2vec, ~10 minutes)

Architecture:
    ┌─────────────────────────────────────────────────┐
    │ Prerequisites (inputs to Stage 6)               │
    │ Stage 2 → qc_h5ad/          — raw count h5ads  │
    │ Stage 3 → rat_to_human_mapping.pickle           │
    │           rat_token_mapping.tsv                 │
    │ Stage 5 → dataset/           — fine-tune corpus │
    │ External → hgnc_complete_set.txt  (auto-dl)     │
    │            mouse_human_orthologs.tsv             │
    └──────────────┬──────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────────┐
    │ Step 1: build_coexp_embedding.py                │
    │ Glob all Stage 2 QC'd h5ad files               │
    │ Per file: sample 3,000 cells (GeneCompass)      │
    │ Validate raw counts                             │
    │ Compute nonzero PCC (both genes ≥ 1)           │
    │ Retain pairs with PCC > 0.8 (GeneCompass)      │
    │ Unify rat gene IDs → human Ensembl             │
    │   (T4 new-token genes keep ENSRNOG)             │
    │ Deduplicate pairs across all studies            │
    │ Train gene2vec Skip-Gram, 768-dim               │
    │ OUT: coexp_gene_pairs.txt                       │
    │      coexp_gene2vec.model                       │
    │      coexp_embeddings.pkl      ← primary        │
    │      stage6_coexp_manifest.json                 │
    └──────────────┬──────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────────┐
    │ Step 2: build_family_embedding.py               │
    │ Load HGNC human gene families (1,645 families)  │
    │ Derive mouse families via BioMart TSV           │
    │ Derive rat families via Stage 3 rat→human       │
    │   (set-valued inversion — all paralogs kept)    │
    │ Generate all within-family gene pairs           │
    │ Unify gene IDs → human Ensembl                 │
    │   (T4 rat genes keep ENSRNOG)                   │
    │ Train gene2vec Skip-Gram, 768-dim               │
    │ OUT: family_gene_pairs.txt                      │
    │      family_gene2vec.model                      │
    │      family_embeddings.pkl     ← primary        │
    │      stage6_family_manifest.json                │
    └──────────────┬──────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────────┐
    │ Downstream consumers (Stage 7)                  │
    │ Fine-tuning → coexp_embeddings.pkl              │
    │               family_embeddings.pkl             │
    │               [promoter_embeddings.pkl — TBD]   │
    │               [grn_embeddings.pkl     — TBD]    │
    └─────────────────────────────────────────────────┘

Timing guidance:
    Step 1 (co-expression) scans 88 studies × ~3,000 sampled cells each.
    PCC computation on ~5,000–10,000 expressed genes per matrix typically
    takes 2–6 hours on a single node. Use --from 2 to re-run Step 2 alone
    after Step 1 completes, without re-running the expensive PCC scan.

    On Gilbreth: Step 1 should be submitted as a standalone SLURM job:
      sbatch --time=8:00:00 --mem=32G --cpus-per-task=8 \
             --wrap="python run_stage6.py --from 1" \
             --job-name=stage6_coexp

Usage:
  python run_stage6.py              # Run both steps
  python run_stage6.py --from 2     # Skip co-expression (Step 1 already done)
  python run_stage6.py --dry-run    # Validate all inputs without running
  python run_stage6.py -v           # Verbose (DEBUG) logging

Author: Tim Reese Lab / Claude
Date: March 2026
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(os.environ.get('PIPELINE_ROOT', '.')).resolve()
sys.path.insert(0, str(_PROJECT_ROOT / 'lib'))
from gene_utils import load_config, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

STAGE_DIR = _PROJECT_ROOT / 'pipeline' / '06_prior_knowledge'

STEPS = [
    {
        'num':         1,
        'name':        'Co-expression embedding',
        'script':      'build_coexp_embedding.py',
        'config_key':  None,               # always enabled (no flag needed)
        'description': (
            'Glob QC h5ads → sample 3,000 cells → nonzero PCC → '
            'pairs > 0.8 → gene2vec 768-dim → coexp_embeddings.pkl'
        ),
        'key_output':  'coexp_embeddings.pkl',
        'timing_note': 'Expected: 2–6 hours (PCC computation over 88 studies)',
    },
    {
        'num':         2,
        'name':        'Gene family embedding',
        'script':      'build_family_embedding.py',
        'config_key':  None,               # always enabled
        'description': (
            'HGNC families → derive mouse/rat → pairwise gene pairs → '
            'gene2vec 768-dim → family_embeddings.pkl'
        ),
        'key_output':  'family_embeddings.pkl',
        'timing_note': 'Expected: ~10 minutes',
    },
    {
        'num':         3,
        'name':        'Promoter sequence embedding',
        'script':      'build_promoter_embedding.py',
        'config_key':  ('prior_knowledge', 'promoter', 'enabled'),
        'description': (
            'BioMart TSS coords + genome FASTA → DNABert [CLS] → '
            'promoter_embeddings.pkl  (use --test for validation run)'
        ),
        'key_output':  'promoter_embeddings.pkl',
        'timing_note': 'Expected: ~4 hours on GPU (full); ~5 min in --test mode',
    },
    {
        'num':         4,
        'name':        'GRN embedding',
        'script':      'build_grn_embedding.py',
        'config_key':  ('prior_knowledge', 'grn', 'enabled'),
        'description': (
            'PECA2vec cross-species transfer from GeneCompass human/mouse GRNs → '
            'grn_embeddings.pkl  (no gene2vec retraining needed)'
        ),
        'key_output':  'grn_embeddings.pkl',
        'timing_note': 'Expected: ~1 minute (dict lookup, no training)',
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_config(config: dict) -> bool:
    """Verify all required config sections and keys are present."""
    errors = []

    for section in ('paths', 'biomart', 'prior_knowledge'):
        if section not in config:
            errors.append(f"Missing config section: '{section}'")

    if errors:
        for e in errors:
            logger.error(e)
        return False

    paths = config['paths']
    required_path_keys = (
        'qc_h5ad_dir',
        'ortholog_dir',
        'prior_knowledge_dir',
        'hgnc_file',
    )
    for key in required_path_keys:
        if key not in paths:
            errors.append(f"Missing paths.{key} in config")

    bm = config.get('biomart', {})
    if 'mouse_human_orthologs' not in bm:
        errors.append("Missing biomart.mouse_human_orthologs in config")

    pk = config.get('prior_knowledge', {})
    if 'coexp' not in pk:
        errors.append("Missing prior_knowledge.coexp section in config")
    if 'gene_family' not in pk:
        errors.append("Missing prior_knowledge.gene_family section in config")
    # Steps 3 and 4 are optional; warn if sections absent but do not error
    for opt_key, step_num in (("promoter", 3), ("grn", 4)):
        if opt_key not in pk:
            logger.warning(
                f"prior_knowledge.{opt_key} section not in config — "
                f"Step {step_num} will be skipped. "
                f"Add '{opt_key}: {{enabled: true}}' to enable it."
            )

    if errors:
        for e in errors:
            logger.error(e)
        return False

    return True


def validate_inputs(config: dict, from_step: int) -> bool:
    """Validate that all inputs required by the requested step(s) are present."""
    errors = []
    paths  = config['paths']

    # ── Inputs always required (both steps need Stage 3) ─────────────────────
    orth_dir         = resolve_path(config, paths['ortholog_dir'])
    rat_to_human     = orth_dir / 'rat_to_human_mapping.pickle'
    rat_token_mapping = orth_dir / 'rat_token_mapping.tsv'

    if not rat_to_human.exists():
        errors.append(
            f"rat_to_human_mapping.pickle not found: {rat_to_human}\n"
            f"  → Run Stage 3 first (python run_stage3.py)"
        )
    else:
        logger.info(
            f"  rat_to_human_mapping.pickle: "
            f"{rat_to_human.stat().st_size / 1e6:.1f} MB [OK]"
        )

    if not rat_token_mapping.exists():
        errors.append(
            f"rat_token_mapping.tsv not found: {rat_token_mapping}\n"
            f"  → Run Stage 3 first (python run_stage3.py)"
        )
    else:
        with open(rat_token_mapping) as f:
            n_genes = sum(1 for _ in f) - 1
        logger.info(f"  rat_token_mapping.tsv: {n_genes:,} rows [OK]")

    # ── Step 1 inputs ─────────────────────────────────────────────────────────
    if from_step <= 1:
        qc_dir     = resolve_path(config, paths['qc_h5ad_dir'])
        h5ad_files = list(qc_dir.glob('**/*.h5ad'))
        if not h5ad_files:
            errors.append(
                f"No .h5ad files found in {qc_dir}\n"
                f"  → Run Stage 2 first (python run_stage2.py)"
            )
        else:
            logger.info(f"  QC'd h5ad files: {len(h5ad_files):,} [OK]")

        # Stage 5 tokenized corpus should exist (confirms we are truly post-S5)
        corpus_dir  = resolve_path(config, paths.get('tokenized_corpus_dir', ''))
        dataset_dir = corpus_dir / 'dataset'
        if corpus_dir and dataset_dir.exists():
            logger.info(f"  Stage 5 dataset/: present [OK]")
        else:
            logger.warning(
                "  Stage 5 dataset/ not found. Stage 6 does not require it, "
                "but its absence suggests the pipeline may not be at Stage 6 yet."
            )

    # ── Step 2 inputs ─────────────────────────────────────────────────────────
    if from_step <= 2:
        bm          = config.get('biomart', {})
        mouse_orth  = resolve_path(config, bm.get('mouse_human_orthologs', ''))
        if not mouse_orth.exists():
            errors.append(
                f"mouse_human_orthologs.tsv not found: {mouse_orth}\n"
                f"  → Download from Ensembl BioMart and place at:\n"
                f"    {mouse_orth}"
            )
        else:
            logger.info(
                f"  mouse_human_orthologs.tsv: "
                f"{mouse_orth.stat().st_size / 1e6:.1f} MB [OK]"
            )

        # HGNC will be auto-downloaded by build_family_embedding.py if missing
        hgnc_path = resolve_path(config, paths.get('hgnc_file', ''))
        if hgnc_path.exists() and hgnc_path.stat().st_size > 10_000:
            logger.info(
                f"  HGNC gene families: "
                f"{hgnc_path.stat().st_size / 1e6:.1f} MB [OK]"
            )
        else:
            logger.info(
                "  HGNC gene families: not found — "
                "build_family_embedding.py will auto-download"
            )

    # ── Step 2 only: check Step 1 outputs ─────────────────────────────────────
    if from_step == 2:
        pk_dir   = resolve_path(config, paths['prior_knowledge_dir'])
        coexp_emb = pk_dir / 'coexp_embeddings.pkl'
        if not coexp_emb.exists():
            errors.append(
                f"coexp_embeddings.pkl not found: {coexp_emb}\n"
                f"  → Run Step 1 first (python run_stage6.py --from 1)\n"
                f"     or:  python run_stage6.py  (runs both steps)"
            )
        else:
            logger.info(
                f"  coexp_embeddings.pkl: "
                f"{coexp_emb.stat().st_size / 1e6:.1f} MB [OK]"
            )

    if errors:
        for e in errors:
            logger.error(e)
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# ENABLED CHECK
# ─────────────────────────────────────────────────────────────────────────────

def is_step_enabled(step: dict, config: dict) -> bool:
    """Return True if this step is enabled in config (or has no config_key)."""
    key_path = step.get('config_key')
    if key_path is None:
        return True  # Steps 1 and 2 are always enabled

    # key_path is a tuple of nested config keys, e.g. ('prior_knowledge', 'promoter', 'enabled')
    node = config
    for k in key_path:
        if not isinstance(node, dict) or k not in node:
            return False
        node = node[k]
    return bool(node)


# ─────────────────────────────────────────────────────────────────────────────
# STEP RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_step(
    step:      dict,
    config:    dict,
    dry_run:   bool = False,
    verbose:   bool = False,
    test_mode: bool = False,
) -> bool:
    """Run one Stage 6 step as a subprocess, mirroring the run_stage4/5 pattern.

    Steps 3 and 4 (promoter, GRN) respect prior_knowledge.<step>.enabled in
    config. If disabled, the step is logged as skipped and True is returned
    so the orchestrator continues normally.

    --test is passed through to Steps 3 and 4 only (coexp and family have no
    test mode — they always run fully).
    """
    # ── Enabled check ─────────────────────────────────────────────────────────
    if not is_step_enabled(step, config):
        logger.info("=" * 70)
        logger.info(f"STEP {step['num']}: {step['name']}  [SKIPPED — disabled in config]")
        logger.info(
            f"  To enable: set prior_knowledge.{step['script'].replace('build_','').replace('_embedding.py','')} "
            f".enabled: true in pipeline_config.yaml"
        )
        logger.info("=" * 70)
        return True

    script_path = STAGE_DIR / step['script']

    if not script_path.exists():
        logger.error(
            f"Script not found: {script_path}\n"
            f"  Expected location: {STAGE_DIR}"
        )
        return False

    cmd = [sys.executable, str(script_path)]
    if dry_run:
        cmd.append('--dry-run')
    if verbose:
        cmd.append('-v')
    # --test only applies to steps 3 and 4 (promoter + GRN have test modes)
    if test_mode and step['num'] in (3, 4):
        cmd.append('--test')

    logger.info("=" * 70)
    logger.info(f"STEP {step['num']}: {step['name']}")
    logger.info(f"  Script:  {script_path.name}")
    logger.info(f"  Action:  {step['description']}")
    logger.info(f"  Timing:  {step['timing_note']}")
    if test_mode and step['num'] in (3, 4):
        logger.info("  Mode:    TEST (limited gene set)")
    logger.info("=" * 70)

    t0     = time.time()
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error(
            f"Step {step['num']} FAILED (exit code {result.returncode}) "
            f"after {elapsed:.1f}s"
        )
        return False

    # Report key output size
    if not dry_run:
        pk_dir  = resolve_path(
            config, config['paths']['prior_knowledge_dir']
        )
        key_out = pk_dir / step['key_output']
        if key_out.exists():
            logger.info(
                f"  Output: {step['key_output']}  "
                f"({key_out.stat().st_size / 1e6:.1f} MB)"
            )

    logger.info(f"Step {step['num']} completed in {elapsed:.1f}s")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Stage 6 Orchestrator: Prior Knowledge Embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1  build_coexp_embedding.py    Nonzero PCC over QC corpus → gene2vec 768-dim
  2  build_family_embedding.py   HGNC gene families → gene2vec 768-dim
  3  build_promoter_embedding.py DNABert on TSS sequences → 768-dim  [requires enabled]
  4  build_grn_embedding.py      TF→TG GRN pairs → gene2vec 768-dim  [requires enabled]

Enable Steps 3 and 4 in pipeline_config.yaml:
  prior_knowledge:
    promoter: {enabled: true}
    grn:      {enabled: true}

Use --test to validate Steps 3/4 on a small gene set before full genome-wide runs.
Step 1 is slow (2–6 hours). Use --from 2 to skip it once complete.

On Gilbreth (recommended for Step 1):
  sbatch --time=8:00:00 --mem=32G --cpus-per-task=8 \\
         --wrap="python run_stage6.py --from 1" \\
         --job-name=stage6_coexp

On Gilbreth (Step 3, GPU recommended for DNABert):
  sbatch --time=4:00:00 --mem=32G --gres=gpu:1 \\
         --wrap="python run_stage6.py --from 3" \\
         --job-name=stage6_promoter

Examples:
  python run_stage6.py              # Run all enabled steps
  python run_stage6.py --from 2     # Skip Step 1 (already done)
  python run_stage6.py --from 3     # Run Steps 3+4 only (test Steps 3/4)
  python run_stage6.py --from 3 --test  # Test mode for Steps 3 and 4
  python run_stage6.py --dry-run    # Validate all inputs without running
  python run_stage6.py -v           # Verbose (DEBUG) logging
        """,
    )
    parser.add_argument(
        '--from', dest='from_step', type=int, default=1,
        choices=[s['num'] for s in STEPS],
        help='Start from this step number (default: 1)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate config and inputs; pass --dry-run to each step script',
    )
    parser.add_argument(
        '--test', action='store_true',
        help=(
            'Pass --test to Steps 3 and 4 (promoter and GRN). '
            'Runs on a small subset to validate the pipeline end-to-end '
            'before committing to a full genome-wide run.'
        ),
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable DEBUG-level logging',
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Config ────────────────────────────────────────────────────────────────
    try:
        config = load_config(_PROJECT_ROOT / 'config' / 'pipeline_config.yaml')
    except FileNotFoundError:
        logger.error(
            "pipeline_config.yaml not found. "
            "Set PIPELINE_ROOT or run from project root."
        )
        sys.exit(1)

    if not validate_config(config):
        logger.error("Config validation failed")
        sys.exit(1)

    if not validate_inputs(config, args.from_step):
        logger.error("Input validation failed")
        sys.exit(1)

    # ── Print step plan ───────────────────────────────────────────────────────
    pk = config.get('prior_knowledge', {})

    logger.info("=" * 70)
    logger.info("STAGE 6: PRIOR KNOWLEDGE EMBEDDINGS")
    logger.info("=" * 70)
    logger.info("")
    for step in STEPS:
        enabled = is_step_enabled(step, config)
        status = "ENABLED" if enabled else "disabled in config"
        test_flag = "  [--test will run in limited mode]" if (args.test and step['num'] in (3, 4) and enabled) else ""
        logger.info(f"  Step {step['num']}: {step['name']:<35s} [{status}]{test_flag}")
    logger.info("")

    coexp = pk.get('coexp', {})
    gf    = pk.get('gene_family', {})
    prom  = pk.get('promoter', {})
    grn   = pk.get('grn', {})

    logger.info(
        f"  coexp:       pcc_threshold={coexp.get('pcc_threshold', 0.8)}, "
        f"n_sample_cells={coexp.get('n_sample_cells', 3000):,}"
    )
    logger.info(
        f"  gene_family: min_family_size={gf.get('min_family_size', 2)}, "
        f"epochs={gf.get('epochs', 30)}"
    )
    if prom.get('enabled'):
        logger.info(
            f"  promoter:    model={prom.get('dnabert_model','zhihan1996/DNA_bert_6')}, "
            f"window=-{prom.get('window_upstream',500)}/+{prom.get('window_downstream',2000)}"
        )
    if grn.get('enabled'):
        logger.info("  grn:         method=transfer (PECA2vec cross-species)")

    if args.from_step > 1:
        logger.info(
            f"\nStarting from Step {args.from_step} "
            f"(skipping Step(s) 1–{args.from_step - 1})"
        )
    if args.test:
        logger.info("\n  TEST MODE active — Steps 3 and 4 will run on limited gene sets")
    logger.info("")

    # ── Run steps ─────────────────────────────────────────────────────────────
    t_total      = time.time()
    steps_to_run = [s for s in STEPS if s['num'] >= args.from_step]
    n_run = n_skipped = 0

    for step in steps_to_run:
        ok = run_step(
            step, config,
            dry_run=args.dry_run,
            verbose=args.verbose,
            test_mode=args.test,
        )
        if not ok:
            logger.error(f"Stage 6 aborted at Step {step['num']}")
            sys.exit(1)
        if is_step_enabled(step, config):
            n_run += 1
        else:
            n_skipped += 1

    elapsed_total = time.time() - t_total

    logger.info("=" * 70)
    logger.info(
        f"STAGE 6 COMPLETE — {n_run} step(s) run, {n_skipped} skipped, "
        f"{elapsed_total:.1f}s total"
    )
    logger.info("=" * 70)

    if not args.dry_run:
        pk_dir = resolve_path(config, config['paths']['prior_knowledge_dir'])
        logger.info(f"\n  Output directory: {pk_dir}")

        all_outputs = [
            'coexp_embeddings.pkl',
            'family_embeddings.pkl',
            'promoter_embeddings.pkl',
            'grn_embeddings.pkl',
        ]
        for name in all_outputs:
            p = pk_dir / name
            if p.exists():
                logger.info(f"  {name:<32s} {p.stat().st_size / 1e6:.1f} MB")

        logger.info("")
        logger.info("Next steps:")
        logger.info("  Stage 7 (fine-tuning):")
        logger.info(
            "    python vendor/GeneCompass/scripts/pretrain.py \\"
        )
        logger.info(
            "      --data_path data/training/tokenized_corpus/dataset \\"
        )
        for name in all_outputs:
            p = pk_dir / name
            if p.exists():
                flag = name.replace('_embeddings.pkl', '')
                logger.info(
                    f"      --{flag}_embeddings "
                    f"data/training/prior_knowledge/{name} \\"
                )

        not_ready = [
            name for name in all_outputs
            if not (pk_dir / name).exists()
        ]
        if not_ready:
            logger.info("")
            logger.info("  Not yet generated:")
            for name in not_ready:
                logger.info(f"    {name}")


if __name__ == '__main__':
    main()