#!/usr/bin/env python3
"""
analyze_singleton_genes.py — Vocabulary Pruning Pipeline for Expanded Corpus

Two-stage vocabulary pruning:
  Stage A: Singleton removal — drop genes appearing in only 1 study
  Stage B: Biotype filter — retain only protein-coding + miRNA
           (matching GeneCompass methodology), using BioMart + RGD fallback

Scans all QC'd h5ad matrices from the expanded preprocessing run,
identifies genes appearing in only one study, classifies biotypes
using BioMart and RGD references, and produces a detailed report.

Outputs:
  - singleton_analysis_report.json   (machine-readable full report)
  - singleton_analysis_summary.txt   (human-readable summary)
  - singleton_genes_to_drop.txt      (gene IDs removed by singleton filter)
  - biotype_genes_to_drop.txt        (gene IDs removed by biotype filter)
  - rat_ensembl_ids_pruned_v3.txt    (after singleton removal only)
  - rat_ensembl_ids_pruned_v4.txt    (final: singletons + biotype filtered)
  - per_study/                        (per-study singleton lists)
  - singleton_gene_details.csv       (per-gene singleton details)

Usage:
  python analyze_singleton_genes.py \
      --qc-dir /depot/reese18/data/training/preprocessed/qc_matrices \
      --pruned-genes /depot/reese18/data/training/preprocessed/rat_ensembl_ids_pruned.txt \
      --biomart-dir /depot/reese18/data/references/biomart \
      --rgd-genes /depot/reese18/data/references/biomart/GENES_RAT.txt \
      --ortholog-dir /depot/reese18/data/training/ortholog_mappings \
      --catalog /depot/reese18/data/catalog/unified_studies.json \
      -o /depot/reese18/data/training/singleton_analysis \
      -v
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path

import h5py
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

# Biotypes to retain (matching GeneCompass preprocess.py)
# Normalized to lowercase for comparison across BioMart and RGD naming
KEEP_BIOTYPES = {
    "protein_coding",   # BioMart convention
    "protein-coding",   # RGD convention
    "mirna",            # RGD convention (lowercase)
}
# BioMart uses "miRNA" (mixed case) — we lowercase everything for matching


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def load_gene_list(path: Path) -> set:
    """Load a newline-delimited gene ID file."""
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def load_biomart(biomart_dir: Path) -> dict:
    """Load BioMart gene info into a dict keyed by Ensembl ID.

    Returns dict of {ensembl_id: {"symbol": ..., "biotype": ..., ...}}
    """
    import pandas as pd
    gi_path = biomart_dir / "rat_gene_info.tsv"
    if not gi_path.exists():
        print(f"  [WARN] BioMart file not found: {gi_path}")
        return {}
    gi = pd.read_csv(gi_path, sep="\t", dtype=str)
    gi.columns = ["ensembl_id", "symbol", "biotype", "chrom", "description"]
    return gi.set_index("ensembl_id").to_dict("index")


def load_rgd_biotypes(rgd_path: Path) -> tuple:
    """Load RGD gene file and extract Ensembl ID -> biotype/symbol mappings.

    RGD ENSEMBL_ID field can be semicolon-separated (one RGD gene -> multiple Ensembl IDs).
    Returns (ens_to_biotype, ens_to_symbol) dicts.
    """
    import pandas as pd

    if not rgd_path.exists():
        print(f"  [WARN] RGD file not found: {rgd_path}")
        return {}, {}

    with open(rgd_path) as f:
        lines = [l for l in f if not l.startswith("#")]

    rgd = pd.read_csv(StringIO("".join(lines)), sep="\t", low_memory=False)

    ens_to_biotype = {}
    ens_to_symbol = {}
    for _, row in rgd.iterrows():
        ens_field = str(row.get("ENSEMBL_ID", ""))
        gene_type = str(row.get("GENE_TYPE", "")).strip()
        symbol = str(row.get("SYMBOL", "")).strip()
        for eid in ens_field.split(";"):
            eid = eid.strip()
            if eid.startswith("ENSRNOG"):
                ens_to_biotype[eid] = gene_type
                ens_to_symbol[eid] = symbol

    return ens_to_biotype, ens_to_symbol


def get_biotype(gene_id: str, biomart: dict, rgd_biotypes: dict) -> tuple:
    """Get biotype and source for a gene, BioMart first then RGD fallback.

    Returns (biotype_str, source_str).
    """
    info = biomart.get(gene_id, {})
    if info:
        return info.get("biotype", ""), "BioMart"

    rgd_bt = rgd_biotypes.get(gene_id, "")
    if rgd_bt and rgd_bt != "nan":
        return rgd_bt, "RGD"

    return "", "Unknown"


def is_kept_biotype(biotype: str) -> bool:
    """Check if a biotype string matches the GC-equivalent keep list."""
    return biotype.lower().strip() in KEEP_BIOTYPES


def load_ortholog_tiers(ortholog_dir: Path) -> dict:
    """Load ortholog mapping and return gene -> tier dict."""
    import pickle
    mapping_path = ortholog_dir / "rat_to_genecompass_mapping.pkl"
    if not mapping_path.exists():
        json_path = ortholog_dir / "rat_to_genecompass_mapping.json"
        if json_path.exists():
            with open(json_path) as f:
                mapping = json.load(f)
        else:
            print(f"  [WARN] No ortholog mapping found in {ortholog_dir}")
            return {}
    else:
        with open(mapping_path, "rb") as f:
            mapping = pickle.load(f)

    gene_tiers = {}
    for gene_id, info in mapping.items():
        if isinstance(info, dict):
            gene_tiers[gene_id] = info.get("tier", "unknown")
        elif isinstance(info, (list, tuple)) and len(info) >= 2:
            gene_tiers[gene_id] = info[1]
        else:
            gene_tiers[gene_id] = "unknown"
    return gene_tiers


def load_catalog(catalog_path: Path) -> dict:
    """Load study catalog for metadata lookups."""
    if not catalog_path.exists():
        return {}
    with open(catalog_path) as f:
        catalog = json.load(f)
    if isinstance(catalog, list):
        return {s.get("accession", s.get("study_id", "")): s for s in catalog}
    return catalog


def read_h5ad_genes(h5path: Path) -> list:
    """Extract gene IDs from an h5ad file."""
    with h5py.File(h5path, "r") as f:
        if "var" not in f:
            return []
        for key in ["_index", "index", "gene_ids"]:
            if key in f["var"]:
                genes = f["var"][key][:]
                return [g.decode() if isinstance(g, bytes) else str(g) for g in genes]
    return []


def read_h5ad_cell_count(h5path: Path) -> int:
    """Get cell count from h5ad file."""
    with h5py.File(h5path, "r") as f:
        if "obs" in f:
            for key in ["_index", "index"]:
                if key in f["obs"]:
                    return len(f["obs"][key])
        if "X" in f:
            if hasattr(f["X"], "shape"):
                return f["X"].shape[0]
            if "shape" in f["X"].attrs:
                return f["X"].attrs["shape"][0]
    return 0


def extract_accession(filename: str) -> str:
    """Extract study accession (GSExxxxx) from h5ad filename."""
    parts = filename.replace(".h5ad", "").split("_")
    for p in parts:
        if p.startswith("GSE") or p.startswith("E-"):
            return p
    return parts[0]


# ──────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────

def analyze_singletons(args):
    t_start = time.time()

    qc_dir = Path(args.qc_dir)
    pruned_genes = load_gene_list(Path(args.pruned_genes))
    biomart = load_biomart(Path(args.biomart_dir)) if args.biomart_dir else {}
    rgd_biotypes, rgd_symbols = load_rgd_biotypes(Path(args.rgd_genes)) if args.rgd_genes else ({}, {})
    gene_tiers = load_ortholog_tiers(Path(args.ortholog_dir)) if args.ortholog_dir else {}
    catalog = load_catalog(Path(args.catalog)) if args.catalog else {}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    verbose = args.verbose

    print(f"References loaded:")
    print(f"  BioMart: {len(biomart):,} genes")
    print(f"  RGD:     {len(rgd_biotypes):,} genes")
    print(f"  Orthologs: {len(gene_tiers):,} genes")

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Scan all QC matrices
    # ══════════════════════════════════════════════════════════════════
    h5_files = sorted(qc_dir.glob("*.h5ad"))
    if not h5_files:
        print(f"ERROR: No h5ad files found in {qc_dir}")
        sys.exit(1)

    print(f"\n[1] Scanning {len(h5_files)} h5ad files in {qc_dir} ...")

    gene_to_studies = defaultdict(set)
    gene_to_samples = defaultdict(int)
    study_genes = defaultdict(set)
    study_cells = defaultdict(int)
    study_samples = defaultdict(int)
    all_genes_in_corpus = set()

    for i, h5path in enumerate(h5_files):
        acc = extract_accession(h5path.name)
        genes = read_h5ad_genes(h5path)
        n_cells = read_h5ad_cell_count(h5path)

        for g in genes:
            gene_to_studies[g].add(acc)
            gene_to_samples[g] += 1
        study_genes[acc].update(genes)
        study_cells[acc] += n_cells
        study_samples[acc] += 1
        all_genes_in_corpus.update(genes)

        if verbose and (i + 1) % 50 == 0:
            print(f"  ... scanned {i + 1}/{len(h5_files)} files")

    n_studies = len(study_genes)
    n_total_cells = sum(study_cells.values())
    print(f"  {len(h5_files)} files across {n_studies} studies, {n_total_cells:,} total cells")
    print(f"  {len(all_genes_in_corpus):,} unique genes in corpus")
    print(f"  {len(pruned_genes):,} genes in pruned vocabulary")

    pruned_in_corpus = pruned_genes & all_genes_in_corpus
    pruned_not_in_corpus = pruned_genes - all_genes_in_corpus
    corpus_not_in_pruned = all_genes_in_corpus - pruned_genes

    print(f"  Pruned genes found in corpus: {len(pruned_in_corpus):,}")
    print(f"  Pruned genes NOT in corpus:   {len(pruned_not_in_corpus):,}")
    print(f"  Corpus genes NOT in pruned:   {len(corpus_not_in_pruned):,}")

    # ══════════════════════════════════════════════════════════════════
    # STAGE A: Singleton Analysis & Removal
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"STAGE A: SINGLETON ANALYSIS")
    print(f"{'='*60}")

    # Gene frequency distribution
    freq_dist = Counter()
    for g in pruned_in_corpus:
        n = len(gene_to_studies[g])
        freq_dist[n] += 1

    singletons = {g for g in pruned_in_corpus if len(gene_to_studies[g]) == 1}
    doubletons = {g for g in pruned_in_corpus if len(gene_to_studies[g]) == 2}
    tripletons = {g for g in pruned_in_corpus if len(gene_to_studies[g]) == 3}
    high_freq = {g for g in pruned_in_corpus if len(gene_to_studies[g]) >= n_studies * 0.5}

    print(f"\n  Gene frequency in pruned vocabulary:")
    print(f"    1 study (singletons):   {len(singletons):>6,}  ({len(singletons)/len(pruned_in_corpus)*100:.1f}%)")
    print(f"    2 studies (doubletons):  {len(doubletons):>6,}  ({len(doubletons)/len(pruned_in_corpus)*100:.1f}%)")
    print(f"    3 studies (tripletons):  {len(tripletons):>6,}  ({len(tripletons)/len(pruned_in_corpus)*100:.1f}%)")
    print(f"    ≥50% of studies:         {len(high_freq):>6,}  ({len(high_freq)/len(pruned_in_corpus)*100:.1f}%)")

    # Singleton breakdown by study
    singleton_by_study = defaultdict(set)
    for g in singletons:
        for acc in gene_to_studies[g]:
            singleton_by_study[acc].add(g)

    print(f"\n  Singleton genes by study (top 20):")
    print(f"  {'Study':15} {'Singletons':>10} {'Total Genes':>12} {'Cells':>10} {'Samples':>8} {'Singleton%':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*10} {'-'*8} {'-'*10}")

    study_singleton_sorted = sorted(singleton_by_study.items(), key=lambda x: -len(x[1]))
    for acc, genes_set in study_singleton_sorted[:20]:
        total_g = len(study_genes[acc])
        cells = study_cells[acc]
        samples = study_samples[acc]
        pct = len(genes_set) / total_g * 100 if total_g else 0
        print(f"  {acc:15} {len(genes_set):>10,} {total_g:>12,} {cells:>10,} {samples:>8} {pct:>9.1f}%")

    # Singleton biotype and tier distributions
    singleton_biotypes = Counter()
    singleton_in_biomart = 0
    singleton_not_in_biomart = 0
    singleton_details = []

    for g in sorted(singletons):
        bt, src = get_biotype(g, biomart, rgd_biotypes)
        if not bt:
            bt = "NOT_ANNOTATED"
        symbol = ""
        info = biomart.get(g, {})
        if info:
            symbol = info.get("symbol", "")
            singleton_in_biomart += 1
        else:
            symbol = rgd_symbols.get(g, "")
            singleton_not_in_biomart += 1

        tier = gene_tiers.get(g, "unmapped")
        singleton_biotypes[bt] += 1
        study_acc = list(gene_to_studies[g])[0]

        singleton_details.append({
            "gene_id": g, "symbol": symbol, "biotype": bt,
            "tier": tier, "study": study_acc, "n_samples": gene_to_samples[g],
        })

    print(f"\n  Singleton biotype distribution:")
    for bt, count in singleton_biotypes.most_common(15):
        print(f"    {bt:30} {count:>6,}  ({count/len(singletons)*100:.1f}%)")

    singleton_tiers = Counter()
    for g in singletons:
        tier = gene_tiers.get(g, "unmapped")
        singleton_tiers[tier] += 1

    print(f"\n  Singleton ortholog tier distribution:")
    for tier, count in singleton_tiers.most_common():
        print(f"    {str(tier):30} {count:>6,}  ({count/len(singletons)*100:.1f}%)")

    # Apply singleton filter
    after_singletons = pruned_in_corpus - singletons
    print(f"\n  STAGE A result: {len(pruned_in_corpus):,} → {len(after_singletons):,}  (−{len(singletons):,} singletons)")

    # ══════════════════════════════════════════════════════════════════
    # STAGE B: Biotype Filter (protein-coding + miRNA only)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"STAGE B: BIOTYPE FILTER (protein-coding + miRNA)")
    print(f"{'='*60}")

    # Classify every gene in after_singletons
    biotype_kept = []
    biotype_dropped = []
    biotype_dist = Counter()
    biotype_source = Counter()
    kept_detail = Counter()

    for g in sorted(after_singletons):
        bt, src = get_biotype(g, biomart, rgd_biotypes)
        bt_lower = bt.lower().strip() if bt else ""
        biotype_dist[bt if bt else "NOT_ANNOTATED"] += 1
        biotype_source[src] += 1

        if is_kept_biotype(bt):
            biotype_kept.append(g)
            kept_detail[f"{src}_{bt}"] += 1
        else:
            biotype_dropped.append(g)

    print(f"\n  Full biotype distribution (post-singleton, BioMart + RGD):")
    print(f"  {'Biotype':35} {'Count':>8} {'Pct':>7} {'Keep?':>6}")
    print(f"  {'-'*35} {'-'*8} {'-'*7} {'-'*6}")
    for bt, n in biotype_dist.most_common():
        keep = "✓" if is_kept_biotype(bt) else "✗"
        print(f"  {bt:35} {n:>8,} {n/len(after_singletons)*100:>6.1f}% {keep:>6}")

    print(f"\n  Annotation source: {dict(biotype_source)}")

    print(f"\n  Kept breakdown:")
    for k, v in sorted(kept_detail.items()):
        print(f"    {k:35} {v:>6,}")

    print(f"\n  STAGE B result: {len(after_singletons):,} → {len(biotype_kept):,}  (−{len(biotype_dropped):,} non-protein-coding/miRNA)")

    # ══════════════════════════════════════════════════════════════════
    # Combined summary
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"COMBINED FILTER SUMMARY")
    print(f"{'='*60}")
    print(f"  Input (pruned v1):        {len(pruned_genes):>8,}")
    print(f"  After singleton removal:  {len(after_singletons):>8,}  (−{len(singletons):,})")
    print(f"  After biotype filter:     {len(biotype_kept):>8,}  (−{len(biotype_dropped):,})")
    print(f"  Total removed:            {len(pruned_genes) - len(biotype_kept):>8,}")

    # Final biotype check
    final_pc = sum(1 for g in biotype_kept
                   if get_biotype(g, biomart, rgd_biotypes)[0].lower().strip()
                   in {"protein_coding", "protein-coding"})
    final_mirna = sum(1 for g in biotype_kept
                      if get_biotype(g, biomart, rgd_biotypes)[0].lower().strip() == "mirna")
    print(f"\n  Final vocabulary composition:")
    print(f"    Protein-coding: {final_pc:>8,}")
    print(f"    miRNA:          {final_mirna:>8,}")
    print(f"    Total:          {len(biotype_kept):>8,}")

    print(f"\n  GeneCompass comparison:")
    print(f"    GC human tokens:  23,113")
    print(f"    GC mouse tokens:  27,443")
    print(f"    Rat (ours):       {len(biotype_kept):,}")

    # ══════════════════════════════════════════════════════════════════
    # Write outputs
    # ══════════════════════════════════════════════════════════════════
    print(f"\n  Writing outputs to {out_dir} ...")

    # ── v3: after singletons only ──
    v3_path = out_dir / "rat_ensembl_ids_pruned_v3.txt"
    with open(v3_path, "w") as f:
        for g in sorted(after_singletons):
            f.write(g + "\n")
    print(f"    ✓ rat_ensembl_ids_pruned_v3.txt ({len(after_singletons):,} genes)")

    # ── v4: after singletons + biotype filter (FINAL) ──
    v4_path = out_dir / "rat_ensembl_ids_pruned_v4.txt"
    with open(v4_path, "w") as f:
        for g in sorted(biotype_kept):
            f.write(g + "\n")
    print(f"    ✓ rat_ensembl_ids_pruned_v4.txt ({len(biotype_kept):,} genes) ← FINAL")

    # ── Singleton drop list ──
    drop_path = out_dir / "singleton_genes_to_drop.txt"
    with open(drop_path, "w") as f:
        for g in sorted(singletons):
            f.write(g + "\n")
    print(f"    ✓ singleton_genes_to_drop.txt ({len(singletons):,} genes)")

    # ── Biotype drop list ──
    bt_drop_path = out_dir / "biotype_genes_to_drop.txt"
    with open(bt_drop_path, "w") as f:
        for g in sorted(biotype_dropped):
            f.write(g + "\n")
    print(f"    ✓ biotype_genes_to_drop.txt ({len(biotype_dropped):,} genes)")

    # ── Singleton gene details CSV ──
    if singleton_details:
        csv_path = out_dir / "singleton_gene_details.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "gene_id", "symbol", "biotype", "tier", "study", "n_samples"
            ])
            writer.writeheader()
            writer.writerows(singleton_details)
        print(f"    ✓ singleton_gene_details.csv ({len(singleton_details):,} rows)")

    # ── Per-study singleton lists ──
    per_study_dir = out_dir / "per_study"
    per_study_dir.mkdir(exist_ok=True)
    n_study_files = 0
    for acc, genes_set in singleton_by_study.items():
        if len(genes_set) >= 10:
            with open(per_study_dir / f"{acc}_singletons.txt", "w") as f:
                for g in sorted(genes_set):
                    bt, _ = get_biotype(g, biomart, rgd_biotypes)
                    symbol = biomart.get(g, {}).get("symbol", rgd_symbols.get(g, ""))
                    f.write(f"{g}\t{symbol}\t{bt}\n")
            n_study_files += 1
    print(f"    ✓ per_study/ ({n_study_files} study files)")

    # ── Full JSON report ──
    report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "qc_dir": str(qc_dir),
            "pruned_genes_file": str(args.pruned_genes),
            "n_h5ad_files": len(h5_files),
            "n_studies": n_studies,
            "n_total_cells": n_total_cells,
            "keep_biotypes": sorted(KEEP_BIOTYPES),
        },
        "gene_counts": {
            "total_unique_in_corpus": len(all_genes_in_corpus),
            "pruned_vocabulary": len(pruned_genes),
            "pruned_in_corpus": len(pruned_in_corpus),
            "pruned_not_in_corpus": len(pruned_not_in_corpus),
            "corpus_not_in_pruned": len(corpus_not_in_pruned),
        },
        "stage_a_singletons": {
            "total_singletons": len(singletons),
            "pct_of_pruned": round(len(singletons) / len(pruned_in_corpus) * 100, 2),
            "in_biomart": singleton_in_biomart,
            "not_in_biomart": singleton_not_in_biomart,
            "biotype_distribution": dict(singleton_biotypes.most_common()),
            "tier_distribution": {str(k): v for k, v in singleton_tiers.most_common()},
            "by_study": {
                acc: {
                    "n_singletons": len(genes_set),
                    "total_genes": len(study_genes[acc]),
                    "total_cells": study_cells[acc],
                    "n_samples": study_samples[acc],
                }
                for acc, genes_set in study_singleton_sorted
            },
            "vocab_after": len(after_singletons),
        },
        "stage_b_biotype": {
            "input_genes": len(after_singletons),
            "kept": len(biotype_kept),
            "dropped": len(biotype_dropped),
            "biotype_distribution": {bt: n for bt, n in biotype_dist.most_common()},
            "annotation_source": dict(biotype_source),
            "kept_detail": dict(kept_detail),
            "final_protein_coding": final_pc,
            "final_mirna": final_mirna,
        },
        "frequency_distribution": {
            "singletons": len(singletons),
            "doubletons": len(doubletons),
            "tripletons": len(tripletons),
            "high_freq_50pct": len(high_freq),
            "full_distribution": {str(k): v for k, v in sorted(freq_dist.items())},
        },
        "final_summary": {
            "input": len(pruned_genes),
            "after_singletons": len(after_singletons),
            "after_biotype": len(biotype_kept),
            "total_removed": len(pruned_genes) - len(biotype_kept),
            "genecompass_human": 23113,
            "genecompass_mouse": 27443,
        },
    }

    with open(out_dir / "singleton_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"    ✓ singleton_analysis_report.json")

    # ── Human-readable summary ──
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("VOCABULARY PRUNING REPORT — EXPANDED CORPUS")
    summary_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("=" * 70)
    summary_lines.append("")
    summary_lines.append(f"Corpus: {len(h5_files)} files, {n_studies} studies, {n_total_cells:,} cells")
    summary_lines.append(f"Input vocabulary: {len(pruned_genes):,} genes")
    summary_lines.append("")
    summary_lines.append("STAGE A: SINGLETON REMOVAL")
    summary_lines.append(f"  Singletons found:  {len(singletons):>6,}  ({len(singletons)/len(pruned_in_corpus)*100:.1f}%)")
    summary_lines.append(f"  Vocabulary:        {len(pruned_in_corpus):,} → {len(after_singletons):,}")
    summary_lines.append("")
    summary_lines.append("  Top singleton-contributing studies:")
    summary_lines.append(f"  {'Study':15} {'Singletons':>10} {'Cells':>10}")
    for acc, genes_set in study_singleton_sorted[:10]:
        cells = study_cells[acc]
        summary_lines.append(f"  {acc:15} {len(genes_set):>10,} {cells:>10,}")
    summary_lines.append("")
    summary_lines.append("STAGE B: BIOTYPE FILTER (protein-coding + miRNA)")
    summary_lines.append(f"  Kept:    {len(biotype_kept):>6,}")
    summary_lines.append(f"  Dropped: {len(biotype_dropped):>6,}")
    summary_lines.append(f"  Vocabulary: {len(after_singletons):,} → {len(biotype_kept):,}")
    summary_lines.append("")
    summary_lines.append("  Dropped biotypes:")
    dropped_bt_dist = Counter()
    for g in biotype_dropped:
        bt, _ = get_biotype(g, biomart, rgd_biotypes)
        dropped_bt_dist[bt if bt else "NOT_ANNOTATED"] += 1
    for bt, n in dropped_bt_dist.most_common(15):
        summary_lines.append(f"    {bt:30} {n:>6,}")
    summary_lines.append("")
    summary_lines.append("COMBINED RESULT")
    summary_lines.append(f"  Input:              {len(pruned_genes):>8,}")
    summary_lines.append(f"  After singletons:   {len(after_singletons):>8,}  (v3)")
    summary_lines.append(f"  After biotype:      {len(biotype_kept):>8,}  (v4 — FINAL)")
    summary_lines.append(f"    Protein-coding:   {final_pc:>8,}")
    summary_lines.append(f"    miRNA:            {final_mirna:>8,}")
    summary_lines.append("")
    summary_lines.append("GENECOMPASS COMPARISON")
    summary_lines.append(f"  GC human tokens:    23,113")
    summary_lines.append(f"  GC mouse tokens:    27,443")
    summary_lines.append(f"  Rat (ours, v4):     {len(biotype_kept):,}")
    summary_lines.append("")
    summary_lines.append("OUTPUT FILES")
    summary_lines.append(f"  rat_ensembl_ids_pruned_v3.txt  — After singletons ({len(after_singletons):,})")
    summary_lines.append(f"  rat_ensembl_ids_pruned_v4.txt  — FINAL vocab ({len(biotype_kept):,})")
    summary_lines.append(f"  singleton_genes_to_drop.txt    — Singleton IDs ({len(singletons):,})")
    summary_lines.append(f"  biotype_genes_to_drop.txt      — Biotype-filtered IDs ({len(biotype_dropped):,})")
    summary_lines.append(f"  singleton_analysis_report.json — Full machine-readable report")
    summary_lines.append("")

    elapsed = time.time() - t_start
    summary_lines.append(f"Completed in {elapsed:.1f}s")

    summary_text = "\n".join(summary_lines)
    print(f"\n{summary_text}")

    with open(out_dir / "singleton_analysis_summary.txt", "w") as f:
        f.write(summary_text + "\n")
    print(f"\n    ✓ singleton_analysis_summary.txt")

    print(f"\nDone. All outputs in {out_dir}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vocabulary pruning: singleton removal + biotype filtering"
    )
    parser.add_argument(
        "--qc-dir", required=True,
        help="Directory containing QC'd h5ad matrices"
    )
    parser.add_argument(
        "--pruned-genes", required=True,
        help="Path to pruned gene list (rat_ensembl_ids_pruned.txt)"
    )
    parser.add_argument(
        "--biomart-dir", default=None,
        help="Directory containing rat_gene_info.tsv from BioMart"
    )
    parser.add_argument(
        "--rgd-genes", default=None,
        help="Path to GENES_RAT.txt from RGD (biotype fallback)"
    )
    parser.add_argument(
        "--ortholog-dir", default=None,
        help="Directory containing ortholog mapping files"
    )
    parser.add_argument(
        "--catalog", default=None,
        help="Path to unified_studies.json catalog"
    )
    parser.add_argument(
        "-o", "--output-dir", required=True,
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print detailed progress"
    )
    args = parser.parse_args()
    analyze_singletons(args)


if __name__ == "__main__":
    main()