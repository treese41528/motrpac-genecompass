#!/usr/bin/env python3
"""
annotation_recovery.py -- Reverse pipeline: reconnect sample-level metadata

Joins four existing data sources to produce a per-sample annotation inventory:

  1. preprocessing_report.json  →  sample_id → source file → GSM accession
  2. {GSE}_samples.tsv          →  GSM → tissue, strain, sex, organism, characteristics
  3. llm_study_analysis.json    →  study-level topic, disease, exercise, MoTrPAC utility
  4. Arrow training corpus      →  study_id → corpus membership, cell counts

Output:
  annotation_inventory.json   - Full structured output
  annotation_inventory.tsv    - Flat table for quick inspection
  tissue_coverage_report.tsv  - Per MoTrPAC tissue: studies, samples, cells, metadata quality

Usage:
  python annotation_recovery.py [--output-dir reports/annotations]

No GPU required. Runs in ~2 min (Arrow scan is the bottleneck).
"""

import argparse
import csv
import gzip
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/depot/reese18/apps/motrpac-genecompass")

TISSUE_MAP_CACHE = PROJECT_ROOT / "reports/annotations/tissue_motrpac_map.json"

# MoTrPAC 18-tissue panel (from Nair et al. 2024 / grant Submission.pdf)
MOTRPAC_TISSUES = [
    "gastrocnemius", "vastus lateralis", "heart", "liver", "kidney",
    "lung", "WAT-SC", "BAT", "adrenal", "blood RNA", "colon",
    "small intestine", "hippocampus", "hypothalamus", "cortex",
    "spleen", "ovary", "testis",
]

# Regex for extracting GSM accession from file paths
GSM_RE = re.compile(r"(GSM\d{6,9})")

# Common GEO characteristics column patterns
CHARACTERISTICS_PATTERNS = {
    "tissue": re.compile(r"tissue|organ|source_name|body.?site", re.I),
    "strain": re.compile(r"strain|genotype|background", re.I),
    "sex": re.compile(r"sex|gender", re.I),
    "age": re.compile(r"age|developmental.?stage|life.?stage", re.I),
    "treatment": re.compile(r"treatment|condition|group|intervention|diet|exercise", re.I),
    "cell_type": re.compile(r"cell.?type|cell.?population|sorted.?population", re.I),
}

# Tissue keywords extractable from GSM filenames or titles
# e.g., GSM4331829_Kidney-M-O → tissue=kidney, sex=M, condition=O(ld)
FILENAME_TISSUE_MAP = {
    "aorta": "aorta", "heart": "heart", "liver": "liver",
    "kidney": "kidney", "lung": "lung", "brain": "brain",
    "cortex": "cortex", "hippocampus": "hippocampus",
    "hypothalamus": "hypothalamus", "muscle": "skeletal_muscle",
    "gastrocnemius": "gastrocnemius", "vastus": "vastus lateralis",
    "bat": "BAT", "wat": "WAT-SC", "adipose": "adipose",
    "spleen": "spleen", "colon": "colon", "intestine": "small intestine",
    "blood": "blood RNA", "adrenal": "adrenal", "ovary": "ovary",
    "testis": "testis", "skin": "skin", "bone": "bone marrow",
    "bm": "bone marrow", "retina": "retina", "pancreas": "pancreas",
}

FILENAME_SEX_MAP = {"-m-": "male", "-f-": "female", "_m_": "male", "_f_": "female"}
FILENAME_CONDITION_MAP = {
    "-y-": "young", "-o-": "old", "-cr-": "caloric_restriction",
    "_y_": "young", "_o_": "old", "_cr_": "caloric_restriction",
    "control": "control", "sham": "sham",
}


# ---------------------------------------------------------------------------
# Step 0: LLM-based tissue → MoTrPAC mapping
# ---------------------------------------------------------------------------
def build_tissue_map_via_llm(unique_tissues: List[str], cache_path: Path) -> Dict[str, Optional[str]]:
    """
    Send unique tissue strings to Claude for mapping to MoTrPAC tissues.

    Returns: {raw_tissue_string: motrpac_tissue_or_null}
    Caches result to cache_path for reuse.
    """
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package required for --build-tissue-map. pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""Map each tissue name to:
1. A normalized tissue name (lowercase, correcting typos, standardizing synonyms)
2. The closest MoTrPAC tissue from the panel below, or null if no reasonable match

MoTrPAC 18-tissue panel:
gastrocnemius, vastus lateralis, heart, liver, kidney, lung, WAT-SC, BAT, adrenal, blood RNA, colon, small intestine, hippocampus, hypothalamus, cortex, spleen, ovary, testis

Rules:
- Correct obvious typos (e.g., "ling" → "lung")
- Normalize case and synonyms (e.g., "Spinal Cord" and "Spinal cord" → "spinal cord")
- Cardiovascular sub-regions (ventricle, atrium, aorta, sinoatrial node, etc.) → motrpac: "heart"
- Pulmonary vasculature → motrpac: "lung"
- Generic "brain" or unspecified brain regions → motrpac: "cortex"
- Limbic structures (amygdala, nucleus accumbens) → motrpac: "hippocampus"
- Basal ganglia (striatum, ventral pallidum) → motrpac: "cortex"
- White/subcutaneous adipose → motrpac: "WAT-SC"
- Brown/interscapular adipose → motrpac: "BAT"
- Perivascular adipose → motrpac: closest depot type
- Generic skeletal muscle → motrpac: "gastrocnemius"
- Bone marrow, PBMC → motrpac: "blood RNA"
- Thymus → motrpac: "spleen"
- Tumors, cell lines, embryos, retina, cartilage, skin, synovium, dorsal root ganglia → motrpac: null (but still normalize the name)
- Proximal jejunum, duodenum, ileum → motrpac: "small intestine"

Tissue names to map:
{json.dumps(sorted(unique_tissues), indent=2)}

Return ONLY a JSON object where each key is the original tissue name and each value is an object with "normalized" and "motrpac" fields. Example:
{{"Left ventricle region 2B": {{"normalized": "left ventricle", "motrpac": "heart"}}, "Retina": {{"normalized": "retina", "motrpac": null}}}}

No markdown, no explanation."""

    logger.info(f"Calling Claude API to map {len(unique_tissues)} tissue values...")
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=4096,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.content[0].text.strip()

    # Strip markdown fences if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    tissue_map = json.loads(content)
    logger.info(f"  Mapped: {sum(1 for v in tissue_map.values() if v)} matched, "
                f"{sum(1 for v in tissue_map.values() if not v)} null")

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "model": "claude-haiku-4-5",
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "n_tissues": len(unique_tissues),
            "mapping": tissue_map,
        }, f, indent=2)
    logger.info(f"  Cached to {cache_path}")

    return tissue_map


def load_tissue_map(cache_path: Path) -> Optional[Dict[str, Optional[str]]]:
    """Load cached tissue mapping if it exists."""
    if not cache_path.exists():
        return None
    try:
        with open(cache_path) as f:
            data = json.load(f)
        mapping = data.get("mapping", {})
        logger.info(f"Step 0: Loaded cached tissue map ({len(mapping)} entries from {data.get('generated_at', '?')})")
        return mapping
    except Exception as e:
        logger.warning(f"Failed to load tissue map cache: {e}")
        return None


# ---------------------------------------------------------------------------
# Step 1: Parse preprocessing_report.json
# ---------------------------------------------------------------------------
def load_preprocessing_report(path: Path) -> Dict[str, Dict]:
    """
    Parse preprocessing_report.json into a dict keyed by sample_id.

    Returns: {sample_id: {accession, source, gsm, format, cells_final, status}}
    """
    with open(path) as f:
        report = json.load(f)

    samples = {}
    for study in report.get("studies", []):
        accession = study["accession"]
        for s in study.get("samples", []):
            sid = s.get("sample_id", "")
            source = s.get("source", "")

            # Extract GSM from source path
            gsm_match = GSM_RE.search(os.path.basename(source))
            gsm = gsm_match.group(1) if gsm_match else None

            samples[sid] = {
                "accession": accession,
                "source_path": source,
                "gsm": gsm,
                "format": s.get("format"),
                "cells_final": s.get("cells_final", 0),
                "cells_raw": s.get("cells_raw", 0),
                "status": s.get("status", "success" if s.get("cells_final") else "failed"),
                "gene_id_format": s.get("gene_id_format"),
            }

    logger.info(f"Step 1: {len(samples)} samples from preprocessing_report.json")
    return samples


# ---------------------------------------------------------------------------
# Step 2: Parse {GSE}_samples.tsv for GSM-level metadata
# ---------------------------------------------------------------------------
def _parse_samples_tsv(filepath: Path) -> Dict[str, Dict]:
    """
    Parse a GEO samples.tsv file into a dict keyed by GSM accession.

    These files are tab-separated with a header row. Column names vary by study
    but commonly include: geo_accession, title, source_name_ch1, organism_ch1,
    and characteristics_ch1.N.{attribute} columns.
    """
    gsm_metadata = {}
    try:
        with open(filepath, "r", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            fieldnames = reader.fieldnames or []

            for row in reader:
                gsm = row.get("geo_accession", "").strip()
                if not gsm.startswith("GSM"):
                    continue

                meta = {
                    "title": row.get("title", ""),
                    "source_name": row.get("source_name_ch1", ""),
                    "organism": row.get("organism_ch1", ""),
                    "platform": row.get("platform_id", ""),
                }

                # Extract all characteristics columns
                characteristics = {}
                for col in fieldnames:
                    if "characteristics_ch1" in col.lower():
                        # Column names like: characteristics_ch1.0.tissue
                        # or: characteristics_ch1.0.strain
                        parts = col.split(".")
                        attr_name = parts[-1] if len(parts) > 2 else col
                        val = row.get(col, "").strip()
                        if val:
                            characteristics[attr_name.lower()] = val

                meta["characteristics"] = characteristics

                # Parse structured fields from characteristics
                for target, pattern in CHARACTERISTICS_PATTERNS.items():
                    for ckey, cval in characteristics.items():
                        if pattern.search(ckey):
                            meta.setdefault(target, cval)

                # Fallback: parse tissue/sex/condition from title or source_name
                title_lower = (meta["title"] + " " + meta["source_name"]).lower()
                if "tissue" not in meta:
                    for keyword, tissue in FILENAME_TISSUE_MAP.items():
                        if keyword in title_lower:
                            meta["tissue"] = tissue
                            break

                gsm_metadata[gsm] = meta

    except Exception as e:
        logger.warning(f"Failed to parse {filepath}: {e}")

    return gsm_metadata


def load_geo_sample_metadata(
    raw_dirs: List[Path], corpus_accessions: set
) -> Dict[str, Dict[str, Dict]]:
    """
    Load _samples.tsv for all corpus studies.

    Returns: {GSE: {GSM: {title, source_name, organism, tissue, strain, sex, ...}}}
    """
    all_metadata = {}

    for raw_dir in raw_dirs:
        if not raw_dir.exists():
            continue
        for study_dir in sorted(raw_dir.iterdir()):
            if not study_dir.is_dir():
                continue
            accession = study_dir.name
            if accession not in corpus_accessions:
                continue

            samples_tsv = study_dir / f"{accession}_samples.tsv"
            if samples_tsv.exists():
                gsm_data = _parse_samples_tsv(samples_tsv)
                if gsm_data:
                    all_metadata[accession] = gsm_data

    logger.info(
        f"Step 2: Loaded samples.tsv for {len(all_metadata)} studies, "
        f"{sum(len(v) for v in all_metadata.values())} total GSMs"
    )
    return all_metadata


# ---------------------------------------------------------------------------
# Step 3: Parse filename-encoded metadata
# ---------------------------------------------------------------------------
def parse_filename_metadata(source_path: str) -> Dict[str, Optional[str]]:
    """
    Extract tissue, sex, condition from GSM-prefixed filenames.

    Examples:
        GSM4331829_Kidney-M-O_matrix.mtx.gz → tissue=kidney, sex=male, condition=old
        GSM5121163_Control_1_matrix.mtx.gz   → condition=control
        GSM4710730_M10_matrix.mtx.gz         → (minimal info)
    """
    fname = os.path.basename(source_path).lower()
    result = {"filename_tissue": None, "filename_sex": None, "filename_condition": None}

    # Tissue from filename
    for keyword, tissue in FILENAME_TISSUE_MAP.items():
        if keyword in fname:
            result["filename_tissue"] = tissue
            break

    # Sex from filename patterns
    for pattern, sex in FILENAME_SEX_MAP.items():
        if pattern in fname:
            result["filename_sex"] = sex
            break

    # Condition from filename patterns
    for pattern, condition in FILENAME_CONDITION_MAP.items():
        if pattern in fname:
            result["filename_condition"] = condition
            break

    return result


# ---------------------------------------------------------------------------
# Step 4: Load LLM study-level analysis
# ---------------------------------------------------------------------------
def load_llm_analysis(path: Path) -> Dict[str, Dict]:
    """
    Load LLM study analysis, keyed by accession.

    Returns study-level metadata: topic, disease, exercise, MoTrPAC utility, etc.
    """
    with open(path) as f:
        data = json.load(f)

    llm_by_acc = {}
    for entry in data.get("analyses", []):
        if not entry.get("_meta", {}).get("success"):
            continue
        acc = entry.get("accession")
        if not acc:
            continue

        organism = entry.get("organism", {})
        if not isinstance(organism, dict):
            organism = {}

        llm_by_acc[acc] = {
            "topic": entry.get("study_overview", {}).get("primary_topic"),
            "topic_category": entry.get("study_overview", {}).get("topic_category"),
            "species": organism.get("species"),
            "strain": organism.get("strain"),
            "sex": organism.get("sex"),
            "age_value": organism.get("age_value"),
            "age_unit": organism.get("age_unit"),
            "life_stage": organism.get("life_stage"),
            "is_single_cell": entry.get("study_type", {}).get("is_single_cell"),
            "is_time_series": entry.get("study_type", {}).get("is_time_series"),
            "is_disease_study": entry.get("study_type", {}).get("is_disease_study"),
            "has_exercise": entry.get("treatments", {}).get("has_exercise"),
            "exercise_type": entry.get("treatments", {}).get("exercise_type"),
            "has_diet": entry.get("treatments", {}).get("has_diet"),
            "diet_type": entry.get("treatments", {}).get("diet_type"),
            "disease_name": entry.get("disease_condition", {}).get("disease_name"),
            "disease_type": entry.get("disease_condition", {}).get("disease_type"),
            "cell_types_mentioned": entry.get("cell_types", []),
            "motrpac_tissues": entry.get("utility_for_motrpac", {}).get("motrpac_tissues", []),
            "deconvolution_useful": entry.get("utility_for_motrpac", {}).get("deconvolution_useful"),
            "grn_useful": entry.get("utility_for_motrpac", {}).get("grn_useful"),
            "tissues": entry.get("tissues", []),
        }

    logger.info(f"Step 4: {len(llm_by_acc)} studies from LLM analysis")
    return llm_by_acc


# ---------------------------------------------------------------------------
# Step 5: Load Arrow corpus for cell counts per study
# ---------------------------------------------------------------------------
def load_corpus_study_counts(corpus_path: Path) -> Dict[str, int]:
    """
    Count cells per study_id in the Arrow training corpus.
    """
    from datasets import load_from_disk

    logger.info(f"Step 5: Loading Arrow corpus from {corpus_path}")
    ds = load_from_disk(str(corpus_path))
    study_ids = ds["study_id"]
    counts = Counter(study_ids)
    logger.info(f"  {len(counts)} unique studies, {sum(counts.values()):,} total cells")
    return dict(counts)


# ---------------------------------------------------------------------------
# Step 6: Join everything
# ---------------------------------------------------------------------------
def build_annotation_inventory(
    samples: Dict[str, Dict],
    geo_metadata: Dict[str, Dict[str, Dict]],
    llm_data: Dict[str, Dict],
    corpus_counts: Dict[str, int],
    tissue_map: Optional[Dict[str, Optional[str]]] = None,
) -> List[Dict]:
    """
    Join all sources into a per-sample annotation record.
    """
    inventory = []

    for sample_id, sample in sorted(samples.items()):
        if sample["status"] != "success" and sample.get("cells_final", 0) == 0:
            continue

        acc = sample["accession"]
        gsm = sample.get("gsm")

        record = {
            "sample_id": sample_id,
            "accession": acc,
            "gsm": gsm,
            "source_path": sample["source_path"],
            "format": sample["format"],
            "cells_final": sample["cells_final"],
            "cells_raw": sample["cells_raw"],
            "gene_id_format": sample.get("gene_id_format"),
            "in_corpus": acc in corpus_counts,
            "corpus_cells_total": corpus_counts.get(acc, 0),
        }

        # --- GEO sample-level metadata (from _samples.tsv) ---
        geo_study = geo_metadata.get(acc, {})
        geo_sample = geo_study.get(gsm, {}) if gsm else {}

        record["geo_title"] = geo_sample.get("title", "")
        record["geo_source_name"] = geo_sample.get("source_name", "")
        record["geo_organism"] = geo_sample.get("organism", "")
        record["geo_platform"] = geo_sample.get("platform", "")
        record["geo_tissue"] = geo_sample.get("tissue")
        record["geo_strain"] = geo_sample.get("strain")
        record["geo_sex"] = geo_sample.get("sex")
        record["geo_age"] = geo_sample.get("age")
        record["geo_treatment"] = geo_sample.get("treatment")
        record["geo_cell_type"] = geo_sample.get("cell_type")
        record["geo_characteristics"] = geo_sample.get("characteristics", {})

        # --- Filename-encoded metadata ---
        fname_meta = parse_filename_metadata(sample["source_path"])
        record.update(fname_meta)

        # --- Resolved metadata (prefer GEO > filename > LLM) ---
        llm = llm_data.get(acc, {})
        llm_tissues = llm.get("motrpac_tissues", [])

        record["tissue_resolved"] = (
            geo_sample.get("tissue")
            or fname_meta.get("filename_tissue")
            or (llm_tissues[0] if llm_tissues else None)
        )
        record["sex_resolved"] = (
            geo_sample.get("sex")
            or fname_meta.get("filename_sex")
            or llm.get("sex")
        )
        record["strain_resolved"] = geo_sample.get("strain") or llm.get("strain")
        record["condition_resolved"] = (
            geo_sample.get("treatment")
            or fname_meta.get("filename_condition")
        )

        # --- MoTrPAC tissue match ---
        tissue_r = (record["tissue_resolved"] or "").strip()
        motrpac_match = None
        tissue_normalized = tissue_r.lower() if tissue_r else None

        # Prefer the LLM-generated tissue map (exact lookup)
        if tissue_map and tissue_r:
            entry = tissue_map.get(tissue_r) or tissue_map.get(tissue_r.lower())
            # Case-insensitive fallback
            if entry is None:
                tissue_r_lower = tissue_r.lower()
                for k, v in tissue_map.items():
                    if k.lower() == tissue_r_lower:
                        entry = v
                        break
            if isinstance(entry, dict):
                tissue_normalized = entry.get("normalized", tissue_r.lower())
                motrpac_match = entry.get("motrpac")
            elif isinstance(entry, str):
                # Backward compat with old format
                motrpac_match = entry

        # Fallback: match against LLM study-level tissue entries
        if motrpac_match is None and tissue_r:
            llm_tissue_entries = llm.get("tissues", [])
            tissue_r_lower = tissue_r.lower()
            best_score = 0
            for lentry in llm_tissue_entries:
                if not isinstance(lentry, dict):
                    continue
                llm_name = (lentry.get("name") or "").lower().strip()
                if not llm_name:
                    continue
                if tissue_r_lower == llm_name:
                    score = 100
                elif llm_name in tissue_r_lower or tissue_r_lower in llm_name:
                    score = 50 + len(llm_name)
                else:
                    words_r = set(tissue_r_lower.split())
                    words_l = set(llm_name.split())
                    overlap = words_r & words_l
                    score = len(overlap) * 10 if overlap else 0
                if score > best_score:
                    best_score = score
                    motrpac_match = lentry.get("motrpac_match")

        record["tissue_normalized"] = tissue_normalized
        record["motrpac_tissue_match"] = motrpac_match

        # --- LLM study-level context ---
        record["llm_topic"] = llm.get("topic")
        record["llm_topic_category"] = llm.get("topic_category")
        record["llm_is_single_cell"] = llm.get("is_single_cell")
        record["llm_is_time_series"] = llm.get("is_time_series")
        record["llm_has_exercise"] = llm.get("has_exercise")
        record["llm_exercise_type"] = llm.get("exercise_type")
        record["llm_has_diet"] = llm.get("has_diet")
        record["llm_disease_name"] = llm.get("disease_name")
        record["llm_disease_type"] = llm.get("disease_type")
        record["llm_cell_types_mentioned"] = llm.get("cell_types_mentioned", [])
        record["llm_deconvolution_useful"] = llm.get("deconvolution_useful")
        record["llm_species"] = llm.get("species")
        record["llm_life_stage"] = llm.get("life_stage")

        inventory.append(record)

    logger.info(f"Step 6: {len(inventory)} annotated samples")
    return inventory


# ---------------------------------------------------------------------------
# Step 7: Generate reports
# ---------------------------------------------------------------------------
def generate_tissue_coverage(inventory: List[Dict]) -> List[Dict]:
    """
    Per MoTrPAC tissue: how many studies, samples, cells, and metadata quality.
    """
    tissue_data = defaultdict(lambda: {
        "studies": set(), "samples": 0, "cells": 0,
        "has_sex": 0, "has_strain": 0, "has_condition": 0,
        "has_cell_type_mention": 0, "has_exercise": 0,
        "sex_breakdown": Counter(), "strain_breakdown": Counter(),
        "condition_breakdown": Counter(),
    })

    for r in inventory:
        mt = r.get("motrpac_tissue_match")
        if not mt:
            continue
        d = tissue_data[mt]
        d["studies"].add(r["accession"])
        d["samples"] += 1
        d["cells"] += r.get("cells_final", 0)
        if r.get("sex_resolved"):
            d["has_sex"] += 1
            d["sex_breakdown"][r["sex_resolved"]] += 1
        if r.get("strain_resolved"):
            d["has_strain"] += 1
            d["strain_breakdown"][r["strain_resolved"]] += 1
        if r.get("condition_resolved"):
            d["has_condition"] += 1
            d["condition_breakdown"][r["condition_resolved"]] += 1
        if r.get("llm_cell_types_mentioned"):
            d["has_cell_type_mention"] += 1
        if r.get("llm_has_exercise"):
            d["has_exercise"] += 1

    rows = []
    for tissue in MOTRPAC_TISSUES:
        d = tissue_data.get(tissue)
        if d is None:
            rows.append({
                "tissue": tissue, "n_studies": 0, "n_samples": 0,
                "n_cells": 0, "pct_has_sex": 0, "pct_has_strain": 0,
                "pct_has_condition": 0, "pct_has_cell_type_mention": 0,
                "n_exercise_samples": 0,
                "sex_breakdown": "", "strain_breakdown": "", "condition_breakdown": "",
            })
        else:
            n = d["samples"]
            rows.append({
                "tissue": tissue,
                "n_studies": len(d["studies"]),
                "n_samples": n,
                "n_cells": d["cells"],
                "pct_has_sex": round(100 * d["has_sex"] / n, 1) if n else 0,
                "pct_has_strain": round(100 * d["has_strain"] / n, 1) if n else 0,
                "pct_has_condition": round(100 * d["has_condition"] / n, 1) if n else 0,
                "pct_has_cell_type_mention": round(100 * d["has_cell_type_mention"] / n, 1) if n else 0,
                "n_exercise_samples": d["has_exercise"],
                "sex_breakdown": "; ".join(f"{k}:{v}" for k, v in d["sex_breakdown"].most_common()),
                "strain_breakdown": "; ".join(f"{k}:{v}" for k, v in d["strain_breakdown"].most_common()),
                "condition_breakdown": "; ".join(f"{k}:{v}" for k, v in d["condition_breakdown"].most_common()),
            })

    return rows


def save_tsv(records: List[Dict], path: Path, fieldnames: List[str] = None):
    """Write a list of dicts to TSV."""
    if not records:
        return
    if fieldnames is None:
        fieldnames = list(records[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for r in records:
            # Flatten lists/dicts for TSV
            flat = {}
            for k, v in r.items():
                if isinstance(v, (list, dict)):
                    flat[k] = json.dumps(v) if v else ""
                elif isinstance(v, set):
                    flat[k] = json.dumps(sorted(v))
                else:
                    flat[k] = v
            writer.writerow(flat)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Annotation recovery: reverse pipeline")
    parser.add_argument("--output-dir", default="reports/annotations", help="Output directory")
    parser.add_argument("--skip-corpus", action="store_true",
                        help="Skip Arrow corpus loading (faster, no cell count join)")
    parser.add_argument("--build-tissue-map", action="store_true",
                        help="Call Claude API to build tissue→MoTrPAC mapping (costs ~$0.01)")
    parser.add_argument("--no-tissue-map", action="store_true",
                        help="Skip tissue map even if cached (use LLM study-level fallback only)")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Preprocessing report ---
    report_path = PROJECT_ROOT / "data/training/gene_universe/preprocessing_report.json"
    samples = load_preprocessing_report(report_path)

    # Get unique corpus accessions
    corpus_accessions = {s["accession"] for s in samples.values() if s.get("cells_final", 0) > 0}
    logger.info(f"  {len(corpus_accessions)} studies with successful samples")

    # --- Step 2: GEO sample-level metadata ---
    raw_dirs = [
        PROJECT_ROOT / "data/raw/geo/single_cell/geo_datasets",
        PROJECT_ROOT / "data/raw/geo/bulk/geo_datasets",
        PROJECT_ROOT / "data/raw/arrayexpress/singlecell/datasets",
        PROJECT_ROOT / "data/raw/arrayexpress/bulk/datasets",
    ]
    geo_metadata = load_geo_sample_metadata(raw_dirs, corpus_accessions)

    # --- Step 3: (filename parsing is done per-sample in build_annotation_inventory)

    # --- Step 4: LLM analysis ---
    llm_path = PROJECT_ROOT / "data/catalog/llm_study_analysis.json"
    llm_data = load_llm_analysis(llm_path)

    # --- Step 5: Arrow corpus ---
    if args.skip_corpus:
        corpus_counts = {acc: -1 for acc in corpus_accessions}
        logger.info("Step 5: Skipped (--skip-corpus)")
    else:
        corpus_path = PROJECT_ROOT / "data/training/tokenized_corpus/dataset"
        corpus_counts = load_corpus_study_counts(corpus_path)

    # --- Step 0 (deferred): Tissue map ---
    tissue_map = None
    if not args.no_tissue_map:
        tissue_map = load_tissue_map(TISSUE_MAP_CACHE)

        if tissue_map is None and args.build_tissue_map:
            # First pass: collect unique tissue values
            temp_inventory = build_annotation_inventory(
                samples, geo_metadata, llm_data, corpus_counts, tissue_map=None
            )
            unique_tissues = sorted({
                r["tissue_resolved"] for r in temp_inventory
                if r.get("tissue_resolved")
            })
            logger.info(f"  {len(unique_tissues)} unique tissue values to map")
            tissue_map = build_tissue_map_via_llm(unique_tissues, TISSUE_MAP_CACHE)
        elif tissue_map is None:
            logger.info("Step 0: No tissue map cached. Run with --build-tissue-map to create one.")

    # --- Step 6: Join ---
    inventory = build_annotation_inventory(
        samples, geo_metadata, llm_data, corpus_counts, tissue_map=tissue_map
    )

    # --- Step 7: Save outputs ---
    logger.info("=" * 60)
    logger.info("Step 7: Saving outputs")
    logger.info("=" * 60)

    # Full JSON
    json_path = output_dir / "annotation_inventory.json"
    with open(json_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "n_samples": len(inventory),
            "n_studies": len({r["accession"] for r in inventory}),
            "n_cells_total": sum(r.get("cells_final", 0) for r in inventory),
            "samples": inventory,
        }, f, indent=2, default=str)
    logger.info(f"  {json_path}")

    # Flat TSV
    tsv_fields = [
        "sample_id", "accession", "gsm", "cells_final", "in_corpus",
        "tissue_resolved", "tissue_normalized", "sex_resolved", "strain_resolved", "condition_resolved",
        "motrpac_tissue_match", "geo_organism", "geo_cell_type",
        "llm_topic", "llm_has_exercise", "llm_disease_name",
        "llm_deconvolution_useful", "llm_cell_types_mentioned",
        "filename_tissue", "filename_sex", "filename_condition",
        "geo_title", "geo_source_name",
    ]
    tsv_path = output_dir / "annotation_inventory.tsv"
    save_tsv(inventory, tsv_path, fieldnames=tsv_fields)
    logger.info(f"  {tsv_path}")

    # Tissue coverage report
    coverage = generate_tissue_coverage(inventory)
    coverage_path = output_dir / "tissue_coverage_report.tsv"
    save_tsv(coverage, coverage_path)
    logger.info(f"  {coverage_path}")

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    n_with_tissue = sum(1 for r in inventory if r.get("tissue_resolved"))
    n_with_sex = sum(1 for r in inventory if r.get("sex_resolved"))
    n_with_strain = sum(1 for r in inventory if r.get("strain_resolved"))
    n_with_condition = sum(1 for r in inventory if r.get("condition_resolved"))
    n_with_motrpac = sum(1 for r in inventory if r.get("motrpac_tissue_match"))
    n_in_corpus = sum(1 for r in inventory if r.get("in_corpus"))

    logger.info(f"  Total samples: {len(inventory)}")
    logger.info(f"  In training corpus: {n_in_corpus}")
    logger.info(f"  With tissue: {n_with_tissue} ({100*n_with_tissue/len(inventory):.0f}%)")
    logger.info(f"  With sex: {n_with_sex} ({100*n_with_sex/len(inventory):.0f}%)")
    logger.info(f"  With strain: {n_with_strain} ({100*n_with_strain/len(inventory):.0f}%)")
    logger.info(f"  With condition: {n_with_condition} ({100*n_with_condition/len(inventory):.0f}%)")
    logger.info(f"  MoTrPAC tissue match: {n_with_motrpac}")

    logger.info(f"\n  Tissue coverage (MoTrPAC panel):")
    for row in coverage:
        if row["n_samples"] > 0:
            logger.info(
                f"    {row['tissue']:20s}  {row['n_studies']:3d} studies  "
                f"{row['n_samples']:4d} samples  {row['n_cells']:>8,} cells"
            )
        else:
            logger.info(f"    {row['tissue']:20s}  -- NO COVERAGE --")

    # Full corpus tissue breakdown (including non-MoTrPAC)
    all_tissues = Counter()
    all_tissue_cells = Counter()
    for r in inventory:
        t = r.get("tissue_normalized") or "(unknown)"
        all_tissues[t] += 1
        all_tissue_cells[t] += r.get("cells_final", 0)
    logger.info(f"\n  Full corpus tissue breakdown ({len(all_tissues)} unique):")
    for t, n in all_tissues.most_common():
        logger.info(f"    {t:40s}  {n:4d} samples  {all_tissue_cells[t]:>8,} cells")


if __name__ == "__main__":
    main()