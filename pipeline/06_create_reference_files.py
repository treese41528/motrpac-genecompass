#!/usr/bin/env python3
"""
create_rat_reference_files.py  (v3 — with RGD duplicate collapse)

Create rat reference files for GeneCompass fine-tuning, matching the format
of existing human/mouse reference files.

Uses only local reference files — no API calls.

KEY CHANGE from v2: Identifies genes where RGD maps the same symbol to both
a canonical BioMart Ensembl ID and one or more additional IDs. The additional
RGD-only IDs are excluded, keeping only the canonical. This reduces vocabulary
by ~2,500 tokens with no loss of biology (canonical ID retains all data).

Annotation sources (layered):
  1. rat_gene_info.tsv      (BioMart: ENSRNOG, symbol, biotype, chromosome)
  2. rat_genes_biomart.tsv  (BioMart: adds NCBI gene IDs)
  3. GENES_RAT.txt          (RGD: fallback for genes missing from BioMart)

Usage: python create_rat_reference_files.py
"""

import csv
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

# ============================================================
# CONFIGURATION
# ============================================================
REFERENCES_DIR = "/depot/reese18/data/references"
BIOMART_DIR = os.path.join(REFERENCES_DIR, "biomart")
ORTHOLOG_DIR = "/depot/reese18/data/training/ortholog_mappings"
GENECOMPASS_SCDATA = "/depot/reese18/apps/GeneCompass/scdata"
OUTPUT_DIR = "/depot/reese18/data/training/rat_reference_files"

# Input files
RAT_GENE_INFO = os.path.join(BIOMART_DIR, "rat_gene_info.tsv")
RAT_GENES_BIOMART = os.path.join(REFERENCES_DIR, "rat_genes_biomart.tsv")
GENES_RAT_RGD = os.path.join(BIOMART_DIR, "GENES_RAT.txt")
RAT_TOKEN_MAPPING = os.path.join(ORTHOLOG_DIR, "rat_token_mapping.tsv")
RAT_GENE_MEDIAN = os.path.join(ORTHOLOG_DIR, "rat_gene_median.pickle")
HUMAN_MOUSE_TOKENS = os.path.join(GENECOMPASS_SCDATA, "dict", "human_mouse_tokens.pickle")
HUMAN_MEDIAN = os.path.join(GENECOMPASS_SCDATA, "dict", "human_gene_median_after_filter.pickle")


# ============================================================
# BIOTYPE NORMALIZATION
# ============================================================
BIOTYPE_MAP = {
    "protein-coding": "protein_coding",
    "protein_coding": "protein_coding",
    "mirna": "miRNA",
    "miRNA": "miRNA",
    "lincrna": "lncRNA",
    "lncrna": "lncRNA",
    "lncRNA": "lncRNA",
    "snorna": "snoRNA",
    "snoRNA": "snoRNA",
    "snrna": "snRNA",
    "snRNA": "snRNA",
    "ncrna": "ncRNA",
    "misc_rna": "misc_RNA",
    "pseudo": "pseudogene",
    "pseudogene": "pseudogene",
    "processed_pseudogene": "processed_pseudogene",
    "gene": "gene",
}


def normalize_biotype(raw):
    """Normalize biotype string to canonical BioMart form."""
    raw = raw.strip()
    return BIOTYPE_MAP.get(raw, raw)


# ============================================================
# STEP 1: Load annotations from local reference files
# ============================================================
def load_annotations():
    """
    Build gene annotation dict from local BioMart and RGD files.
    Primary: rat_gene_info.tsv (has chromosome for MT identification)
    Supplement: rat_genes_biomart.tsv (has NCBI ID)
    Fallback: GENES_RAT.txt (RGD, semicolon-delimited Ensembl IDs)
    """
    print("=" * 60)
    print("STEP 1: Loading annotations from local reference files")
    print("=" * 60)

    annotations = {}

    # --- Primary: rat_gene_info.tsv ---
    print(f"\nReading {RAT_GENE_INFO}...")
    n_info = 0
    with open(RAT_GENE_INFO) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        print(f"  Columns: {header}")
        for row in reader:
            if len(row) < 4:
                continue
            gene_id = row[0].strip()
            if not gene_id.startswith("ENSRNOG"):
                continue
            annotations[gene_id] = {
                "symbol": row[1].strip() if row[1].strip() else gene_id,
                "biotype": normalize_biotype(row[2]),
                "chromosome": row[3].strip(),
                "description": row[4].strip() if len(row) > 4 else "",
                "ncbi_id": "",
                "source": "biomart_info",
            }
            n_info += 1
    print(f"  Loaded {n_info} genes from rat_gene_info.tsv")

    # --- Supplement: rat_genes_biomart.tsv (add NCBI IDs) ---
    print(f"\nReading {RAT_GENES_BIOMART}...")
    n_ncbi = 0
    n_new = 0
    with open(RAT_GENES_BIOMART) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        print(f"  Columns: {header}")
        for row in reader:
            if len(row) < 4:
                continue
            gene_id = row[0].strip()
            if not gene_id.startswith("ENSRNOG"):
                continue
            ncbi_id = row[2].strip() if len(row) > 2 else ""

            if gene_id in annotations:
                if ncbi_id:
                    annotations[gene_id]["ncbi_id"] = ncbi_id
                    n_ncbi += 1
            else:
                annotations[gene_id] = {
                    "symbol": row[1].strip() if row[1].strip() else gene_id,
                    "biotype": normalize_biotype(row[3]),
                    "chromosome": "",
                    "description": "",
                    "ncbi_id": ncbi_id,
                    "source": "biomart_genes",
                }
                n_new += 1
    print(f"  Added NCBI IDs for {n_ncbi} genes, {n_new} new genes")

    # --- Fallback: GENES_RAT.txt (RGD) ---
    print(f"\nReading {GENES_RAT_RGD}...")
    n_rgd = 0
    n_rgd_multi = 0
    with open(GENES_RAT_RGD) as f:
        header_line = None
        for line in f:
            if line.startswith("#"):
                continue
            header_line = line.strip()
            break

        if header_line:
            rgd_cols = header_line.split("\t")
            print(f"  RGD columns ({len(rgd_cols)})")

            col_idx = {}
            for i, col in enumerate(rgd_cols):
                name = col.strip()
                if name == "SYMBOL":
                    col_idx["symbol"] = i
                elif name == "GENE_TYPE":
                    col_idx["biotype"] = i
                elif name == "ENSEMBL_ID":
                    col_idx["ensembl"] = i
                elif name == "CHROMOSOME_mRatBN7.2":
                    col_idx["chromosome"] = i
                elif name == "CHROMOSOME_CELERA" and "chromosome" not in col_idx:
                    col_idx["chromosome"] = i

            print(f"  Column indices: {col_idx}")

            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                fields = line.strip().split("\t")

                ensembl_raw = ""
                if "ensembl" in col_idx and col_idx["ensembl"] < len(fields):
                    ensembl_raw = fields[col_idx["ensembl"]].strip()

                if not ensembl_raw:
                    continue

                ensembl_ids = [eid.strip() for eid in ensembl_raw.split(";")
                               if eid.strip().startswith("ENSRNOG")]

                if not ensembl_ids:
                    continue
                if len(ensembl_ids) > 1:
                    n_rgd_multi += 1

                symbol = ""
                if "symbol" in col_idx and col_idx["symbol"] < len(fields):
                    symbol = fields[col_idx["symbol"]].strip()
                biotype = ""
                if "biotype" in col_idx and col_idx["biotype"] < len(fields):
                    biotype = fields[col_idx["biotype"]].strip()
                chrom = ""
                if "chromosome" in col_idx and col_idx["chromosome"] < len(fields):
                    chrom = fields[col_idx["chromosome"]].strip()

                for ensembl_id in ensembl_ids:
                    if ensembl_id in annotations:
                        continue
                    annotations[ensembl_id] = {
                        "symbol": symbol if symbol else ensembl_id,
                        "biotype": normalize_biotype(biotype) if biotype else "unknown_RGD",
                        "chromosome": chrom,
                        "description": "",
                        "ncbi_id": "",
                        "source": "RGD",
                    }
                    n_rgd += 1

    print(f"  Added {n_rgd} new genes from RGD ({n_rgd_multi} entries had multiple Ensembl IDs)")

    # Summary
    print(f"\nTotal annotations: {len(annotations)} genes")
    source_counts = defaultdict(int)
    for v in annotations.values():
        source_counts[v["source"]] += 1
    print(f"Annotation sources: {dict(source_counts)}")

    return annotations


# ============================================================
# STEP 2: Build exclusion list (RGD-only duplicates)
# ============================================================
def build_exclusion_list(annotations):
    """
    Identify RGD-only Ensembl IDs that duplicate a canonical BioMart ID
    (same gene symbol, BioMart has exactly 1 ID, RGD adds extras).
    These are excluded from all downstream files.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Building exclusion list (RGD duplicate collapse)")
    print("=" * 60)

    # Load our full gene set
    all_genes = []
    with open(RAT_TOKEN_MAPPING) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            all_genes.append(row["rat_gene_id"])

    # BioMart gene set (canonical IDs)
    biomart_genes = set()
    with open(RAT_GENE_INFO) as f:
        next(f)
        for line in f:
            biomart_genes.add(line.split("\t")[0].strip())

    # Group our genes by symbol
    symbols = defaultdict(list)
    for gene_id in all_genes:
        if gene_id in annotations:
            sym = annotations[gene_id]["symbol"]
            symbols[sym].append(gene_id)

    # Find collapsible cases: 1 BioMart canonical + N RGD-only extras
    exclude_set = set()
    collapse_log = []

    for sym, ids in symbols.items():
        if len(ids) < 2:
            continue
        bm_ids = [g for g in ids if g in biomart_genes]
        rgd_ids = [g for g in ids if g not in biomart_genes]

        if len(bm_ids) == 1 and len(rgd_ids) >= 1:
            canonical = bm_ids[0]
            for rgd_id in rgd_ids:
                exclude_set.add(rgd_id)
                collapse_log.append({
                    "excluded_id": rgd_id,
                    "canonical_id": canonical,
                    "symbol": sym,
                })

    print(f"Exclusion list: {len(exclude_set)} RGD-only duplicate IDs")
    print(f"  Mapping to {len(set(e['canonical_id'] for e in collapse_log))} canonical BioMart IDs")

    # Save exclusion list
    excl_path = os.path.join(OUTPUT_DIR, "excluded_rgd_duplicates.tsv")
    with open(excl_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["excluded_id", "canonical_id", "symbol"],
                                delimiter="\t")
        writer.writeheader()
        writer.writerows(sorted(collapse_log, key=lambda x: x["symbol"]))
    print(f"Saved: {excl_path}")

    # Remaining gene set
    kept_genes = [g for g in all_genes if g not in exclude_set]
    print(f"\nGene set: {len(all_genes)} -> {len(kept_genes)} (dropped {len(exclude_set)})")

    return exclude_set, kept_genes


# ============================================================
# STEP 3: Match annotations to pruned gene set
# ============================================================
def match_to_genes(annotations, kept_genes):
    """Match annotations and report biotype distribution."""
    print("\n" + "=" * 60)
    print("STEP 3: Matching annotations to pruned gene set")
    print("=" * 60)

    print(f"Pruned gene set: {len(kept_genes)} genes")

    matched = sum(1 for g in kept_genes if g in annotations)
    unmatched = [g for g in kept_genes if g not in annotations]

    print(f"Matched: {matched}/{len(kept_genes)} ({100*matched/len(kept_genes):.1f}%)")
    if unmatched:
        print(f"Unmatched: {len(unmatched)}")

    our_biotypes = defaultdict(int)
    for gene_id in kept_genes:
        if gene_id in annotations:
            our_biotypes[annotations[gene_id]["biotype"]] += 1
        else:
            our_biotypes["NOT_ANNOTATED"] += 1

    print(f"\nBiotype distribution in {len(kept_genes)} genes:")
    for bt, count in sorted(our_biotypes.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(kept_genes)
        print(f"  {bt}: {count:,} ({pct:.1f}%)")

    return kept_genes


# ============================================================
# STEP 4: Create gene list files
# ============================================================
def create_gene_lists(annotations, kept_genes):
    """Create rat_protein_coding.txt, rat_miRNA.txt, rat_mitochondria."""
    print("\n" + "=" * 60)
    print("STEP 4: Creating gene list files")
    print("=" * 60)

    protein_coding = []
    mirna = []
    mitochondrial = []

    for gene_id in kept_genes:
        if gene_id not in annotations:
            continue
        ann = annotations[gene_id]
        symbol = ann["symbol"]
        biotype = ann["biotype"]
        chrom = ann["chromosome"]

        if biotype == "protein_coding":
            protein_coding.append(symbol)
        elif biotype == "miRNA":
            mirna.append(symbol)
        if chrom == "MT":
            mitochondrial.append(symbol)

    protein_coding.sort()
    mirna.sort()
    mitochondrial.sort()

    pc_path = os.path.join(OUTPUT_DIR, "rat_protein_coding.txt")
    with open(pc_path, "w") as f:
        for sym in protein_coding:
            f.write(f"{sym}\tprotein_coding\n")
    print(f"rat_protein_coding.txt: {len(protein_coding)} genes")

    mirna_path = os.path.join(OUTPUT_DIR, "rat_miRNA.txt")
    with open(mirna_path, "w") as f:
        for sym in mirna:
            f.write(f"{sym}\tmiRNA\n")
    print(f"rat_miRNA.txt: {len(mirna)} genes")

    try:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.append(["Chromosome", "Gene name"])
        for sym in mitochondrial:
            ws.append(["chrMT", sym])
        mito_path = os.path.join(OUTPUT_DIR, "rat_mitochondria.xlsx")
        wb.save(mito_path)
        print(f"rat_mitochondria.xlsx: {len(mitochondrial)} genes")
    except ImportError:
        print("openpyxl not available, writing TSV fallback...")
        mito_path = os.path.join(OUTPUT_DIR, "rat_mitochondria.tsv")
        with open(mito_path, "w") as f:
            f.write("Chromosome\tGene name\n")
            for sym in mitochondrial:
                f.write(f"chrMT\t{sym}\n")
        print(f"rat_mitochondria.tsv (xlsx fallback): {len(mitochondrial)} genes")

    filtered_genes = set()
    for gene_id in kept_genes:
        if gene_id in annotations:
            bt = annotations[gene_id]["biotype"]
            if bt in ("protein_coding", "miRNA"):
                filtered_genes.add(gene_id)

    print(f"\nFiltered set (protein_coding + miRNA): {len(filtered_genes)}/{len(kept_genes)} "
          f"({100*len(filtered_genes)/len(kept_genes):.1f}%)")

    unannotated = [g for g in kept_genes if g not in annotations]
    if unannotated:
        unannotated_path = os.path.join(OUTPUT_DIR, "unannotated_genes.txt")
        with open(unannotated_path, "w") as f:
            for g in sorted(unannotated):
                f.write(g + "\n")
        print(f"WARNING: {len(unannotated)} unannotated -> unannotated_genes.txt")

    return protein_coding, mirna, mitochondrial, filtered_genes


# ============================================================
# STEP 5: Create gene ID mapping dictionaries
# ============================================================
def create_gene_id_dicts(annotations, kept_genes):
    """Create {ENSRNOG -> symbol} and {symbol -> ENSRNOG} dicts."""
    print("\n" + "=" * 60)
    print("STEP 5: Creating gene ID mapping dictionaries")
    print("=" * 60)

    id_to_name = {}
    name_to_id = {}
    dup_symbols = defaultdict(list)

    for gene_id in kept_genes:
        symbol = annotations[gene_id]["symbol"] if gene_id in annotations else gene_id
        id_to_name[gene_id] = symbol
        dup_symbols[symbol].append(gene_id)
        if symbol not in name_to_id:
            name_to_id[symbol] = gene_id

    path1 = os.path.join(OUTPUT_DIR, "Gene_id_name_dict_rat.pickle")
    with open(path1, "wb") as f:
        pickle.dump(id_to_name, f)
    print(f"Gene_id_name_dict_rat.pickle: {len(id_to_name)} entries (ENSRNOG -> symbol)")

    path2 = os.path.join(OUTPUT_DIR, "Gene_id_name_dict1_rat.pickle")
    with open(path2, "wb") as f:
        pickle.dump(name_to_id, f)
    print(f"Gene_id_name_dict1_rat.pickle: {len(name_to_id)} entries (symbol -> ENSRNOG)")

    dups = {s: ids for s, ids in dup_symbols.items() if len(ids) > 1}
    print(f"\n  Remaining duplicate symbols: {len(dups)} (real paralogs)")
    if dups:
        for sym, ids in sorted(dups.items(), key=lambda x: -len(x[1]))[:5]:
            print(f"    {sym}: {len(ids)} IDs")

    return id_to_name, name_to_id


# ============================================================
# STEP 6: Create extended token dictionary
# ============================================================
def create_extended_token_dict(kept_genes, exclude_set):
    """Extended token dict excluding dropped genes."""
    print("\n" + "=" * 60)
    print("STEP 6: Creating extended token dictionary")
    print("=" * 60)

    with open(HUMAN_MOUSE_TOKENS, "rb") as f:
        existing_tokens = pickle.load(f)
    max_existing = max(existing_tokens.values())
    print(f"Existing human_mouse tokens: {len(existing_tokens)} (range 0-{max_existing})")

    # Load rat token mapping, skip excluded
    tier_counts = defaultdict(int)
    rat_gene_token = {}
    new_tokens = {}

    kept_set = set(kept_genes)
    n_skipped = 0

    with open(RAT_TOKEN_MAPPING) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene_id = row["rat_gene_id"]
            if gene_id in exclude_set:
                n_skipped += 1
                continue

            token_id = int(row["token_id"])
            tier = row["tier"]
            rat_gene_token[gene_id] = token_id
            tier_counts[tier] += 1
            if tier in ("mouse-rat", "new"):
                new_tokens[gene_id] = token_id

    print(f"Skipped {n_skipped} excluded genes")
    print(f"\nRat gene tier distribution (after exclusion):")
    for tier, count in sorted(tier_counts.items(), key=lambda x: -x[1]):
        print(f"  {tier}: {count:,}")

    extended = dict(existing_tokens)
    added = 0
    for gene_id, token_id in new_tokens.items():
        if gene_id not in extended:
            extended[gene_id] = token_id
            added += 1

    max_extended = max(extended.values())
    print(f"\nNew tokens added: {added}")
    print(f"Extended token dict: {len(extended)} entries (range 0-{max_extended})")

    path1 = os.path.join(OUTPUT_DIR, "extended_tokens.pickle")
    with open(path1, "wb") as f:
        pickle.dump(extended, f)
    print(f"Saved: {path1}")

    path2 = os.path.join(OUTPUT_DIR, "rat_gene_token_dict.pickle")
    with open(path2, "wb") as f:
        pickle.dump(rat_gene_token, f)
    print(f"rat_gene_token_dict.pickle: {len(rat_gene_token)} entries")

    return extended, rat_gene_token


# ============================================================
# STEP 7: Prepare median dict
# ============================================================
def prepare_median_dict(kept_genes, exclude_set):
    """Copy rat medians, excluding dropped genes."""
    print("\n" + "=" * 60)
    print("STEP 7: Preparing median dictionaries")
    print("=" * 60)

    with open(RAT_GENE_MEDIAN, "rb") as f:
        rat_medians_full = pickle.load(f)

    kept_set = set(kept_genes)
    rat_medians = {k: v for k, v in rat_medians_full.items() if k in kept_set}
    print(f"Rat gene medians: {len(rat_medians_full)} -> {len(rat_medians)} (dropped {len(rat_medians_full) - len(rat_medians)})")

    out_path = os.path.join(OUTPUT_DIR, "rat_gene_median_after_filter.pickle")
    with open(out_path, "wb") as f:
        pickle.dump(rat_medians, f)
    print(f"Saved: {out_path}")

    with open(HUMAN_MEDIAN, "rb") as f:
        human_medians = pickle.load(f)

    rat_vals = list(rat_medians.values())
    human_vals = list(human_medians.values())

    print(f"\n{'Metric':<30} {'Rat':>12} {'Human':>12}")
    print("-" * 54)
    print(f"{'Count':<30} {len(rat_vals):>12,} {len(human_vals):>12,}")
    print(f"{'Min':<30} {min(rat_vals):>12.2f} {min(human_vals):>12.2f}")
    print(f"{'Max':<30} {max(rat_vals):>12.2f} {max(human_vals):>12.2f}")
    print(f"{'Mean':<30} {sum(rat_vals)/len(rat_vals):>12.4f} "
          f"{sum(human_vals)/len(human_vals):>12.4f}")

    n_int_rat = sum(1 for v in rat_vals if v == int(v))
    n_int_human = sum(1 for v in human_vals if v == int(v))
    print(f"{'Integer values':<30} "
          f"{n_int_rat:>8} ({100*n_int_rat/len(rat_vals):.1f}%)  "
          f"{n_int_human:>5} ({100*n_int_human/len(human_vals):.1f}%)")

    rat_median_dist = Counter(rat_vals)
    print(f"\nRat median value distribution:")
    for val, count in sorted(rat_median_dist.items()):
        print(f"  median={val}: {count:,} genes ({100*count/len(rat_vals):.1f}%)")

    return rat_medians


# ============================================================
# STEP 8: Create annotated token mapping
# ============================================================
def create_annotated_mapping(annotations, kept_genes, exclude_set):
    """Extend rat_token_mapping.tsv with biotype, excluding dropped genes."""
    print("\n" + "=" * 60)
    print("STEP 8: Creating annotated token mapping")
    print("=" * 60)

    rows = []
    with open(RAT_TOKEN_MAPPING) as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = list(reader.fieldnames)
        for row in reader:
            if row["rat_gene_id"] in exclude_set:
                continue
            rows.append(dict(row))

    new_cols = ["gene_symbol", "gene_biotype", "chromosome"]
    extended_fieldnames = fieldnames + new_cols

    for row in rows:
        gene_id = row["rat_gene_id"]
        if gene_id in annotations:
            ann = annotations[gene_id]
            row["gene_symbol"] = ann["symbol"]
            row["gene_biotype"] = ann["biotype"]
            row["chromosome"] = ann["chromosome"]
        else:
            row["gene_symbol"] = ""
            row["gene_biotype"] = "NOT_ANNOTATED"
            row["chromosome"] = ""

    out_path = os.path.join(OUTPUT_DIR, "rat_token_mapping_annotated.tsv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=extended_fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_path} ({len(rows)} genes)")

    biotype_counts = defaultdict(int)
    for row in rows:
        biotype_counts[row["gene_biotype"]] += 1

    print(f"\nBiotype distribution:")
    for bt, count in sorted(biotype_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(rows)
        print(f"  {bt}: {count:,} ({pct:.1f}%)")

    n_pc = biotype_counts.get("protein_coding", 0)
    n_mirna = biotype_counts.get("miRNA", 0)
    print(f"\nGeneCompass-filtered set: {n_pc + n_mirna:,} genes "
          f"(protein_coding={n_pc:,}, miRNA={n_mirna:,})")

    return biotype_counts


# ============================================================
# STEP 9: Summary
# ============================================================
def create_summary(annotations, kept_genes, exclude_set, filtered_genes,
                   protein_coding, mirna, mitochondrial,
                   extended_tokens, biotype_counts):
    """Write summary JSON and cross-species comparison."""
    print("\n" + "=" * 60)
    print("STEP 9: Summary & Cross-Species Comparison")
    print("=" * 60)

    summary = {
        "original_gene_count": len(kept_genes) + len(exclude_set),
        "excluded_rgd_duplicates": len(exclude_set),
        "final_gene_count": len(kept_genes),
        "annotated_genes": sum(1 for g in kept_genes if g in annotations),
        "protein_coding_count": len(protein_coding),
        "miRNA_count": len(mirna),
        "mitochondrial_count": len(mitochondrial),
        "filtered_set_count": len(filtered_genes),
        "extended_token_count": len(extended_tokens),
        "biotype_distribution": {k: v for k, v in sorted(biotype_counts.items(), key=lambda x: -x[1])},
        "files_created": [
            "rat_protein_coding.txt",
            "rat_miRNA.txt",
            "rat_mitochondria.xlsx/.tsv",
            "Gene_id_name_dict_rat.pickle",
            "Gene_id_name_dict1_rat.pickle",
            "rat_gene_median_after_filter.pickle",
            "extended_tokens.pickle",
            "rat_gene_token_dict.pickle",
            "rat_token_mapping_annotated.tsv",
            "excluded_rgd_duplicates.tsv",
            "rat_reference_summary.json",
        ],
        "genecompass_comparison": {
            "human_protein_coding": 19362,
            "human_miRNA": 1852,
            "human_mitochondrial": 38,
            "human_median_genes": 19560,
            "human_mouse_tokens": 50558,
            "mouse_protein_coding": 21833,
            "mouse_miRNA": 2201,
        },
    }

    path = os.path.join(OUTPUT_DIR, "rat_reference_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {path}")

    print(f"\n{'=' * 70}")
    print(f"  Cross-Species Reference File Comparison")
    print(f"{'=' * 70}")
    print(f"{'Category':<30} {'Human':>10} {'Mouse':>10} {'Rat':>10}")
    print(f"{'-' * 70}")
    print(f"{'protein_coding genes':<30} {'19,362':>10} {'21,833':>10} "
          f"{len(protein_coding):>10,}")
    print(f"{'miRNA genes':<30} {'1,852':>10} {'2,201':>10} "
          f"{len(mirna):>10,}")
    print(f"{'mitochondrial genes':<30} {'38':>10} {'--':>10} "
          f"{len(mitochondrial):>10}")
    print(f"{'median dict entries':<30} {'19,560':>10} {'(shared)':>10} "
          f"{len(kept_genes):>10,}")
    print(f"{'token dict entries':<30} {'50,558':>10} {'(shared)':>10} "
          f"{len(extended_tokens):>10,}")
    print(f"{'-' * 70}")
    print(f"\n  RGD duplicates excluded: {len(exclude_set)}")
    print(f"  Original gene count:     {len(kept_genes) + len(exclude_set)}")
    print(f"  Final gene count:        {len(kept_genes)}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  Creating Rat Reference Files for GeneCompass")
    print("  v3: With RGD duplicate collapse")
    print("=" * 60)

    required_files = [
        RAT_GENE_INFO, RAT_GENES_BIOMART, GENES_RAT_RGD,
        RAT_TOKEN_MAPPING, RAT_GENE_MEDIAN,
        HUMAN_MOUSE_TOKENS, HUMAN_MEDIAN,
    ]
    for fpath in required_files:
        if not os.path.exists(fpath):
            print(f"ERROR: Required file not found: {fpath}")
            sys.exit(1)
    print("All required input files verified.\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Pipeline
    annotations = load_annotations()
    exclude_set, kept_genes = build_exclusion_list(annotations)
    match_to_genes(annotations, kept_genes)
    protein_coding, mirna, mitochondrial, filtered_genes = \
        create_gene_lists(annotations, kept_genes)
    create_gene_id_dicts(annotations, kept_genes)
    extended_tokens, rat_token_dict = create_extended_token_dict(kept_genes, exclude_set)
    prepare_median_dict(kept_genes, exclude_set)
    biotype_counts = create_annotated_mapping(annotations, kept_genes, exclude_set)
    create_summary(annotations, kept_genes, exclude_set, filtered_genes,
                   protein_coding, mirna, mitochondrial,
                   extended_tokens, biotype_counts)

    print(f"\n{'=' * 60}")
    print("  DONE. All reference files written to:")
    print(f"  {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()