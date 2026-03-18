#!/usr/bin/env python3
"""
build_promoter_embedding.py — Stage 6, Step 3: Promoter Sequence Embedding

Pipeline position:
    Stage 6, Step 1: build_coexp_embedding.py
    Stage 6, Step 2: build_family_embedding.py
    Stage 6, Step 3: build_promoter_embedding.py   ← THIS SCRIPT
    Stage 6, Step 4: build_grn_embedding.py

Status: IN DEVELOPMENT — test mode available.

    Run --test to validate the full pipeline on a small gene set before
    committing to a genome-wide run (~22,000 rat genes × 2,500 bp each).

Purpose:
    Compute 768-dimensional promoter sequence embeddings for rat gene tokens
    following the GeneCompass prior knowledge construction protocol:

      1. For each rat gene in rat_token_mapping.tsv, retrieve a 2,500 bp
         window centred on its annotated TSS from the Ensembl rat genome
         assembly (mRatBN7.2):
           - 500 bp upstream of TSS   (proximal promoter)
           - 2,000 bp downstream of TSS (5' UTR + transcription initiation)

      2. Pass each sequence through DNABert (Ji et al., Bioinformatics 2021),
         a BERT model pre-trained on the human reference genome using k-mer
         tokenization (k=3–6). Extract the [CLS] token from the final hidden
         layer as the 768-dimensional promoter embedding for that gene.

         GeneCompass fine-tuned DNABert for 40 epochs on human+mouse promoter
         sequences. We apply the fine-tuned model directly to rat sequences
         (cross-species transfer), or optionally re-fine-tune including rat.

      3. For ortholog-mapped rat genes (T1–T3), the embedding is stored under
         their unified human Ensembl ID so it aligns with the token lookup.
         T4 genes (new tokens) are stored under their ENSRNOG ID.

      4. Genes with no TSS annotation or sequences with > 10% ambiguous
         nucleotides (Ns) are flagged and excluded from the output.

Design decisions:
    - TSS coordinates are fetched from Ensembl BioMart (the same release used
      in Stage 2, pinned in config: biomart.ensembl_release = "113").
    - DNABert model weights are loaded from HuggingFace Hub unless a local
      path is supplied via config (prior_knowledge.promoter.dnabert_model).
    - The script processes genes in configurable batches for GPU efficiency.
    - In --test mode, only the first N genes (default 200) are processed.
      This validates sequence retrieval, DNABert inference, and output format
      without the multi-hour full run.

Inputs (all paths from pipeline_config.yaml):
    Stage 3 → ortholog_dir/rat_token_mapping.tsv        — gene list + tiers
    Stage 3 → ortholog_dir/rat_to_human_mapping.pickle  — gene ID unification
    BioMart → fetched at runtime for TSS coordinates
    Config  → prior_knowledge.promoter.*

Outputs (all to paths.prior_knowledge_dir):
    promoter_sequences.fa             — FASTA of extracted promoter windows
    promoter_embeddings.pkl           — {gene_id: np.ndarray(768,)} — primary output
    promoter_skipped.tsv              — Genes excluded (no TSS / high N-content)
    stage6_promoter_manifest.json     — Provenance record

Config additions required in pipeline_config.yaml:
    prior_knowledge:
      promoter:
        enabled: true
        dnabert_model: "zhihan1996/DNA_bert_6"   # HuggingFace model ID or local path
        window_upstream: 500                      # bp upstream of TSS
        window_downstream: 2000                   # bp downstream of TSS
        max_n_fraction: 0.10                      # exclude sequences with > 10% Ns
        batch_size: 32                            # sequences per DNABert batch
        fine_tune_on_rat: false                   # re-fine-tune DNABert on rat promoters
        n_test_genes: 200                         # genes to process in --test mode
    paths:
      # Rat genome FASTA is large (~2.9 GB uncompressed). If pre-downloaded,
      # set this path. Otherwise the script will download it from Ensembl FTP.
      rat_genome_fasta: data/references/genome/Rattus_norvegicus.mRatBN7.2.dna.toplevel.fa.gz

Usage:
    # Validate pipeline on 200 genes first
    python pipeline/06_prior_knowledge/build_promoter_embedding.py --test

    # Validate without running any inference
    python pipeline/06_prior_knowledge/build_promoter_embedding.py --dry-run

    # Full genome-wide run (submit as SLURM job — GPU recommended)
    python pipeline/06_prior_knowledge/build_promoter_embedding.py

    # Resume from FASTA (sequences already extracted, skip BioMart/genome step)
    python pipeline/06_prior_knowledge/build_promoter_embedding.py --from-fasta

Author: Tim Reese Lab / Claude
Date: March 2026
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import pickle
import gzip
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Optional heavy imports — checked at runtime, not import time
# ─────────────────────────────────────────────────────────────────────────────
_MISSING_REQUIRED = []
_MISSING_OPTIONAL = []

try:
    import requests as _requests
except ImportError:
    _MISSING_OPTIONAL.append('requests')
    _requests = None

# DNABert / HuggingFace — checked only when actually running inference
_TRANSFORMERS_AVAILABLE = None  # lazy-checked

# ─────────────────────────────────────────────────────────────────────────────
# Project path bootstrap
# ─────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(os.environ.get('PIPELINE_ROOT', '.')).resolve()
sys.path.insert(0, str(_PROJECT_ROOT / 'lib'))
from gene_utils import load_config, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# Ensembl FTP URL for rat genome FASTA (mRatBN7.2, release 113)
_ENSEMBL_GENOME_FTP = (
    'https://ftp.ensembl.org/pub/release-113/fasta/rattus_norvegicus/dna/'
    'Rattus_norvegicus.mRatBN7.2.dna.toplevel.fa.gz'
)

# BioMart REST endpoint for TSS coordinate retrieval
_BIOMART_REST = 'https://www.ensembl.org/biomart/martservice'


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: ENVIRONMENT CHECK
# ═════════════════════════════════════════════════════════════════════════════

def check_transformers() -> bool:
    """Check that transformers and torch are available for DNABert inference."""
    global _TRANSFORMERS_AVAILABLE
    if _TRANSFORMERS_AVAILABLE is not None:
        return _TRANSFORMERS_AVAILABLE

    missing = []
    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append('transformers')
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append('torch')

    if missing:
        logger.error(
            f"DNABert inference requires: {', '.join(missing)}\n"
            f"  Install:  pip install transformers torch\n"
            f"  On Gilbreth (GPU node):  module load cuda/11.8; pip install torch --index-url https://download.pytorch.org/whl/cu118"
        )
        _TRANSFORMERS_AVAILABLE = False
    else:
        _TRANSFORMERS_AVAILABLE = True
    return _TRANSFORMERS_AVAILABLE


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: GENE LIST LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_gene_list(
    ortholog_dir: Path,
    n_test: Optional[int] = None,
) -> Tuple[List[Dict], Dict[str, str]]:
    """Load rat genes from Stage 3 rat_token_mapping.tsv.

    Args:
        ortholog_dir: Stage 3 output directory
        n_test:       if set, return only the first n_test genes (test mode)

    Returns:
        genes:        list of {rat_gene, rat_symbol, biotype, tier, human_ortholog}
        rat_to_human: {rat_ensrnog: human_ensg} for ID unification
    """
    mapping_path = ortholog_dir / 'rat_token_mapping.tsv'
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"rat_token_mapping.tsv not found: {mapping_path}\n"
            f"  → Run Stage 3 first:  python run_stage3.py"
        )

    rat_to_human_path = ortholog_dir / 'rat_to_human_mapping.pickle'
    if not rat_to_human_path.exists():
        raise FileNotFoundError(
            f"rat_to_human_mapping.pickle not found: {rat_to_human_path}"
        )
    with open(rat_to_human_path, 'rb') as f:
        rat_to_human = pickle.load(f)

    genes = []
    with open(mapping_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row.get('tier', '') == 'excluded':
                continue
            genes.append({
                'rat_gene':       row['rat_gene'].strip(),
                'rat_symbol':     row.get('rat_symbol', '').strip(),
                'biotype':        row.get('biotype', '').strip(),
                'tier':           row.get('tier', '').strip(),
                'human_ortholog': row.get('human_ortholog', '').strip(),
            })

    if n_test is not None:
        genes = genes[:n_test]
        logger.info(f"Test mode: using first {len(genes)} genes")

    logger.info(f"Genes to embed: {len(genes):,}")
    return genes, rat_to_human


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: TSS COORDINATE RETRIEVAL
# ═════════════════════════════════════════════════════════════════════════════

def fetch_tss_coordinates(
    ensembl_ids: List[str],
    ensembl_release: str,
    cache_path: Path,
) -> Dict[str, Dict]:
    """Retrieve TSS coordinates from Ensembl BioMart for a list of ENSRNOG IDs.

    Returns the TSS position (transcription start site), chromosome, and
    strand for each gene. Uses the same Ensembl release pinned in Stage 2
    config (biomart.ensembl_release) for reproducibility.

    Results are cached to cache_path to avoid re-fetching on reruns.

    Returns:
        {ensrnog_id: {'chrom': str, 'tss': int, 'strand': int (1 or -1)}}

    TSS is defined as:
        strand +1 → tss = transcript_start
        strand -1 → tss = transcript_end   (coordinates are on + strand)
    """
    if cache_path.exists():
        logger.info(f"  Loading cached TSS coordinates: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    logger.info(f"  Fetching TSS coordinates from Ensembl BioMart (release {ensembl_release})")
    logger.info(f"  Genes to query: {len(ensembl_ids):,}")

    tss_coords: Dict[str, Dict] = {}

    # BioMart XML query — fetches gene_id, chromosome, start, end, strand
    # for all provided Ensembl gene IDs.
    # Batch in groups of 500 to avoid URL length limits.
    batch_size = 500
    n_fetched = 0

    for i in range(0, len(ensembl_ids), batch_size):
        batch = ensembl_ids[i:i + batch_size]

        xml_query = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1" count="" datasetConfigVersion="0.6">
  <Dataset name="rnorvegicus_gene_ensembl" interface="default">
    <Filter name="ensembl_gene_id" value="{','.join(batch)}"/>
    <Attribute name="ensembl_gene_id"/>
    <Attribute name="chromosome_name"/>
    <Attribute name="transcript_start"/>
    <Attribute name="transcript_end"/>
    <Attribute name="strand"/>
  </Dataset>
</Query>"""

        try:
            if _requests is None:
                import urllib.request
                import urllib.parse
                data = urllib.parse.urlencode({'query': xml_query}).encode()
                with urllib.request.urlopen(_BIOMART_REST, data=data, timeout=120) as resp:
                    content = resp.read().decode('utf-8')
            else:
                resp = _requests.post(
                    _BIOMART_REST,
                    data={'query': xml_query},
                    timeout=120,
                )
                resp.raise_for_status()
                content = resp.text

            for line in content.strip().split('\n'):
                if not line or '\t' not in line:
                    continue
                parts = line.split('\t')
                if len(parts) < 5:
                    continue
                gene_id, chrom, t_start, t_end, strand_str = parts[:5]
                try:
                    strand = int(strand_str)
                    tss = int(t_start) if strand == 1 else int(t_end)
                    # Keep only the first (canonical) TSS per gene
                    if gene_id not in tss_coords:
                        tss_coords[gene_id] = {
                            'chrom': chrom,
                            'tss': tss,
                            'strand': strand,
                        }
                except ValueError:
                    continue

            n_fetched += len(batch)
            logger.info(
                f"  Fetched {n_fetched}/{len(ensembl_ids)} genes — "
                f"{len(tss_coords):,} with TSS so far"
            )

        except Exception as exc:
            logger.error(f"  BioMart query failed for batch {i//batch_size + 1}: {exc}")

    logger.info(f"  TSS coordinates retrieved: {len(tss_coords):,}/{len(ensembl_ids):,}")

    # Cache for reuse
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(tss_coords, f, protocol=4)
    logger.info(f"  TSS coordinates cached: {cache_path}")

    return tss_coords


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: GENOME SEQUENCE EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════
def ensure_genome_fasta(genome_fasta_path: Path) -> Path:
    """Ensure the rat genome FASTA exists in plain uncompressed form.

    Accepts .fa or .fa.gz path from config. Logic:
      1. .fa present -> use it directly.
      2. .fa.gz present -> decompress with Python gzip, delete .gz.
      3. Neither -> download .fa.gz from Ensembl FTP, then decompress.

    pyfaidx does not support standard gzip. This function always produces
    a plain .fa so no samtools/bgzip is needed.
    """
    fa_path = Path(str(genome_fasta_path).removesuffix('.gz'))
    gz_path = Path(str(fa_path) + '.gz')

    # Case 1: uncompressed already present
    if fa_path.exists() and fa_path.stat().st_size > 1_000_000:
        logger.info(
            f"  Genome FASTA: {fa_path} "
            f"({fa_path.stat().st_size / 1e9:.2f} GB)"
        )
        return fa_path

    # Case 2: compressed present -- decompress
    if gz_path.exists() and gz_path.stat().st_size > 1_000_000:
        logger.info(
            f"  Decompressing {gz_path.name} -> {fa_path.name} "
            f"(~10 GB output, ~5 minutes) ..."
        )
        with gzip.open(gz_path, 'rb') as f_in, open(fa_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out, length=65536)
        gz_path.unlink()
        logger.info(
            f"  Decompressed: {fa_path.stat().st_size / 1e9:.2f} GB  |  .gz deleted"
        )
        return fa_path

    # Case 3: neither -- download then decompress
    logger.info(f"  Genome FASTA not found -- downloading from Ensembl FTP")
    logger.info(f"  URL: {_ENSEMBL_GENOME_FTP}")
    logger.warning(
        "  Download is ~2.9 GB (~15-20 minutes). "
        "Set paths.rat_genome_fasta to a pre-downloaded .fa to skip this."
    )

    fa_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if _requests is not None:
            with _requests.get(_ENSEMBL_GENOME_FTP, stream=True, timeout=600) as resp:
                resp.raise_for_status()
                with open(gz_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
        else:
            import urllib.request
            urllib.request.urlretrieve(_ENSEMBL_GENOME_FTP, gz_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download genome FASTA: {exc}\n"
            f"  Manual:  nohup wget -c {_ENSEMBL_GENOME_FTP} -P {fa_path.parent} &\n"
            f"  Then re-run -- decompression is automatic."
        )

    logger.info(f"  Downloaded: {gz_path.stat().st_size / 1e9:.2f} GB")
    logger.info(f"  Decompressing -> {fa_path.name} (~5 minutes) ...")
    with gzip.open(gz_path, 'rb') as f_in, open(fa_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out, length=65536)
    gz_path.unlink()
    logger.info(
        f"  Decompressed: {fa_path.stat().st_size / 1e9:.2f} GB  |  .gz deleted"
    )
    return fa_path


def extract_promoter_sequences(
    genes: List[Dict],
    tss_coords: Dict[str, Dict],
    genome_fasta_path: Path,
    upstream_bp: int,
    downstream_bp: int,
    max_n_fraction: float,
) -> Tuple[Dict[str, str], List[Dict]]:
    """Extract promoter sequences for each gene from the genome FASTA.

    For each gene, extracts:
        [TSS - upstream_bp, TSS + downstream_bp]   (+ strand)
        [TSS - downstream_bp, TSS + upstream_bp]   (- strand, reverse-complemented)

    Genes excluded if:
        - No TSS coordinate in tss_coords
        - Chromosome not found in FASTA
        - Sequence has > max_n_fraction ambiguous nucleotides

    Requires pyfaidx for indexed FASTA access (pip install pyfaidx).

    Returns:
        sequences: {rat_ensrnog: sequence_string (2500 bp)}
        skipped:   list of {rat_gene, reason} dicts
    """
    try:
        from pyfaidx import Fasta
    except ImportError:
        raise ImportError(
            "pyfaidx is required for genome sequence extraction:\n"
            "  pip install pyfaidx\n"
            "  After installing, also run:  faidx <genome.fa.gz>"
        )

    logger.info(f"  Opening genome FASTA: {genome_fasta_path}")
    genome = Fasta(str(genome_fasta_path), rebuild=True)

    sequences: Dict[str, str] = {}
    skipped: List[Dict] = []
    window_size = upstream_bp + downstream_bp

    _RC_TABLE = str.maketrans('ACGTacgt', 'TGCAtgca')

    for gene in genes:
        rat_id = gene['rat_gene']

        if rat_id not in tss_coords:
            skipped.append({'rat_gene': rat_id, 'reason': 'no_tss_coordinate'})
            continue

        coord = tss_coords[rat_id]
        chrom  = coord['chrom']
        tss    = coord['tss']
        strand = coord['strand']

        if strand == 1:
            seq_start = max(1, tss - upstream_bp)
            seq_end   = tss + downstream_bp
        else:
            seq_start = max(1, tss - downstream_bp)
            seq_end   = tss + upstream_bp

        try:
            seq = str(genome[chrom][seq_start - 1:seq_end]).upper()
        except (KeyError, ValueError):
            skipped.append({'rat_gene': rat_id, 'reason': f'chrom_not_found:{chrom}'})
            continue

        if strand == -1:
            seq = seq.translate(_RC_TABLE)[::-1]

        # Pad with Ns if sequence is shorter than window (near chromosome ends)
        if len(seq) < window_size:
            seq = seq.ljust(window_size, 'N')

        n_fraction = seq.count('N') / len(seq)
        if n_fraction > max_n_fraction:
            skipped.append({
                'rat_gene': rat_id,
                'reason': f'high_n_fraction:{n_fraction:.3f}',
            })
            continue

        sequences[rat_id] = seq

    logger.info(
        f"  Sequences extracted: {len(sequences):,} | "
        f"Skipped: {len(skipped):,}"
    )
    return sequences, skipped


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: DNABERT INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

def build_promoter_embeddings(
    sequences: Dict[str, str],
    rat_to_human: Dict[str, str],
    dnabert_model_name: str,
    batch_size: int,
    kmer: int = 6,
) -> Dict[str, np.ndarray]:
    """Run DNABert inference on promoter sequences to produce 768-dim embeddings.

    Uses the pretrained DNABert model (fine-tuned by GeneCompass on human+mouse
    promoters) as a frozen encoder. The [CLS] token output of the final hidden
    layer is used as the gene's promoter embedding — a 768-dimensional vector
    summarizing the full 2,500 bp promoter window.

    GeneCompass protocol:
        - DNABert pretrained on human genome (hg19), k-mer k=6 tokenization
        - Fine-tuned for 40 epochs on human+mouse promoter sequences
        - [CLS] token → 768-dim embedding

    Args:
        sequences:         {rat_ensrnog: sequence_string}
        rat_to_human:      Stage 3 rat→human mapping (for ID unification)
        dnabert_model_name: HuggingFace model ID or local path
        batch_size:        sequences per batch
        kmer:              k-mer size for tokenization (DNABert default: 6)

    Returns:
        {unified_gene_id: np.ndarray(768,)}
        Keys use human Ensembl IDs for ortholog-mapped genes (T1–T3),
        ENSRNOG IDs for T4 genes.
    """
    if not check_transformers():
        raise ImportError("transformers and torch are required for DNABert inference")

    import torch
    from transformers import AutoTokenizer, AutoModel

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"  DNABert inference on: {device}")
    if device == 'cpu':
        logger.warning(
            "  Running on CPU — this will be slow for large gene sets. "
            "On Gilbreth, request a GPU node:  --gres=gpu:1"
        )

    logger.info(f"  Loading DNABert model: {dnabert_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(dnabert_model_name)
    model     = AutoModel.from_pretrained(dnabert_model_name)
    model.to(device)
    model.eval()

    def seq_to_kmers(seq: str, k: int = 6) -> str:
        """Convert raw DNA sequence to space-separated k-mers for DNABert."""
        return ' '.join(seq[i:i+k] for i in range(0, len(seq) - k + 1))

    gene_ids = list(sequences.keys())
    embeddings: Dict[str, np.ndarray] = {}

    logger.info(
        f"  Running inference on {len(gene_ids):,} sequences "
        f"(batch_size={batch_size})"
    )

    with torch.no_grad():
        for i in range(0, len(gene_ids), batch_size):
            batch_ids   = gene_ids[i:i + batch_size]
            batch_seqs  = [seq_to_kmers(sequences[gid], kmer) for gid in batch_ids]

            inputs = tokenizer(
                batch_seqs,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            # [CLS] token is index 0 of last_hidden_state
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            for j, rat_id in enumerate(batch_ids):
                # Unify: ortholog-mapped → human Ensembl ID, T4 → ENSRNOG
                unified_id = rat_to_human.get(rat_id, rat_id)
                embeddings[unified_id] = cls_embeddings[j].astype(np.float32)

            if (i // batch_size + 1) % 10 == 0:
                logger.info(
                    f"  Batches complete: {i // batch_size + 1} / "
                    f"{(len(gene_ids) + batch_size - 1) // batch_size}"
                )

    logger.info(f"  Embeddings generated: {len(embeddings):,}")
    return embeddings


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: MANIFEST
# ═════════════════════════════════════════════════════════════════════════════

def file_md5(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    output_dir: Path,
    config: dict,
    n_genes_input: int,
    n_sequences_extracted: int,
    n_skipped: int,
    n_embeddings: int,
    vector_size: int,
    test_mode: bool,
    elapsed: float,
    dry_run: bool,
    status: str,
) -> None:
    pk   = config.get('prior_knowledge', {})
    prom = pk.get('promoter', {})

    manifest = {
        'stage':   '6_step3_promoter_embedding',
        'script':  'build_promoter_embedding.py',
        'generated': datetime.utcnow().isoformat() + 'Z',
        'status':  status,
        'dry_run': dry_run,
        'test_mode': test_mode,
        'elapsed_seconds': round(elapsed, 1),
        'parameters': {
            'window_upstream':   prom.get('window_upstream', 500),
            'window_downstream': prom.get('window_downstream', 2000),
            'window_total_bp':   prom.get('window_upstream', 500) + prom.get('window_downstream', 2000),
            'max_n_fraction':    prom.get('max_n_fraction', 0.10),
            'batch_size':        prom.get('batch_size', 32),
            'dnabert_model':     prom.get('dnabert_model', 'zhihan1996/DNA_bert_6'),
            'kmer':              6,
            'n_test_genes':      prom.get('n_test_genes', 200) if test_mode else None,
        },
        'outputs': {
            'genes_input':         n_genes_input,
            'sequences_extracted': n_sequences_extracted,
            'genes_skipped':       n_skipped,
            'embeddings':          n_embeddings,
            'embedding_dimension': vector_size,
        },
        'config_snapshot': {
            'biomart': config.get('biomart', {}),
            'prior_knowledge.promoter': prom,
        },
    }

    manifest_path = output_dir / 'stage6_promoter_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written: {manifest_path}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Stage 6 Step 3: Promoter Sequence Embedding (DNABert)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Recommended workflow:
  1. Validate environment and inputs:
       python build_promoter_embedding.py --dry-run

  2. Run on first 200 genes to verify sequence extraction and DNABert inference:
       python build_promoter_embedding.py --test

  3. Review test outputs in prior_knowledge/ directory, then run full genome-wide:
       python build_promoter_embedding.py

  4. Resume from existing FASTA (skip BioMart + genome extraction):
       python build_promoter_embedding.py --from-fasta

On Gilbreth (GPU recommended for DNABert inference):
  sbatch --time=4:00:00 --mem=32G --gres=gpu:1 \\
         --wrap="python run_stage6.py --from 3"
        """,
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate all inputs and config without running inference',
    )
    parser.add_argument(
        '--test', action='store_true',
        help=(
            'Run on first N genes only (prior_knowledge.promoter.n_test_genes). '
            'Use this to validate the pipeline before a full genome-wide run.'
        ),
    )
    parser.add_argument(
        '--from-fasta', action='store_true',
        help=(
            'Skip TSS retrieval and genome extraction. '
            'Load existing promoter_sequences.fa from output directory.'
        ),
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable DEBUG-level logging',
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    t_start = time.time()

    # ── Load configuration ────────────────────────────────────────────────────
    config    = load_config(_PROJECT_ROOT / 'config' / 'pipeline_config.yaml')
    paths_cfg = config.get('paths', {})
    pk_cfg    = config.get('prior_knowledge', {})
    prom_cfg  = pk_cfg.get('promoter', {})
    bm_cfg    = config.get('biomart', {})

    # ── Resolve paths ─────────────────────────────────────────────────────────
    ortholog_dir  = resolve_path(config, paths_cfg['ortholog_dir'])
    output_dir    = resolve_path(config, paths_cfg['prior_knowledge_dir'])
    genome_fasta  = resolve_path(
        config,
        paths_cfg.get('rat_genome_fasta',
                      'data/references/genome/Rattus_norvegicus.mRatBN7.2.dna.toplevel.fa.gz'),
    )

    # ── Parameters ────────────────────────────────────────────────────────────
    upstream_bp   = int(prom_cfg.get('window_upstream',   500))
    downstream_bp = int(prom_cfg.get('window_downstream', 2000))
    max_n_frac    = float(prom_cfg.get('max_n_fraction',  0.10))
    batch_size    = int(prom_cfg.get('batch_size',        32))
    dnabert_model = str(prom_cfg.get('dnabert_model',     'zhihan1996/DNA_bert_6'))
    n_test_genes  = int(prom_cfg.get('n_test_genes',      200)) if args.test else None
    vector_size   = 768  # Fixed — DNABert hidden size

    logger.info("=" * 60)
    logger.info("Stage 6 Step 3: Promoter Sequence Embedding")
    logger.info("=" * 60)
    logger.info(f"  Project root:   {_PROJECT_ROOT}")
    logger.info(f"  Output dir:     {output_dir}")
    logger.info(f"  DNABert model:  {dnabert_model}")
    logger.info(f"  Window:         -{upstream_bp} / +{downstream_bp} bp around TSS")
    logger.info(f"  Batch size:     {batch_size}")
    logger.info(f"  Test mode:      {args.test}{f' ({n_test_genes} genes)' if args.test else ''}")
    logger.info(f"  Dry run:        {args.dry_run}")

    # ── Validate Stage 3 prerequisites ───────────────────────────────────────
    for fname in ('rat_token_mapping.tsv', 'rat_to_human_mapping.pickle'):
        p = ortholog_dir / fname
        if not p.exists():
            logger.error(f"MISSING: {fname} — {p}")
            logger.error("  → Run Stage 3 first:  python run_stage3.py")
            sys.exit(1)
        logger.info(f"  {fname}: {p.stat().st_size / 1e6:.1f} MB [OK]")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        logger.info("Dry run complete — all inputs validated.")
        write_manifest(
            output_dir, config,
            n_genes_input=0, n_sequences_extracted=0,
            n_skipped=0, n_embeddings=0, vector_size=vector_size,
            test_mode=args.test, elapsed=time.time() - t_start,
            dry_run=True, status='dry_run',
        )
        return

    # ── Check DNABert dependencies ────────────────────────────────────────────
    if not check_transformers():
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 1: Load gene list
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\nPhase 1: Loading gene list from Stage 3")
    logger.info("-" * 60)
    genes, rat_to_human = load_gene_list(ortholog_dir, n_test=n_test_genes)

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 2: Sequence extraction
    # ═══════════════════════════════════════════════════════════════════════════
    fasta_path = output_dir / 'promoter_sequences.fa'

    if args.from_fasta and fasta_path.exists():
        logger.info(f"\nPhase 2: Loading existing FASTA: {fasta_path}")
        sequences: Dict[str, str] = {}
        with open(fasta_path) as f:
            current_id = None
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    current_id = line[1:].split()[0]
                elif current_id:
                    sequences[current_id] = sequences.get(current_id, '') + line
        skipped: List[Dict] = []
        logger.info(f"  Sequences loaded: {len(sequences):,}")

    else:
        logger.info("\nPhase 2: Retrieving TSS coordinates and extracting sequences")
        logger.info("-" * 60)

        rat_ids = [g['rat_gene'] for g in genes]
        tss_cache = output_dir / 'tss_coordinates_cache.pkl'
        tss_coords = fetch_tss_coordinates(rat_ids, bm_cfg.get('ensembl_release', '113'), tss_cache)

        genome_fasta = ensure_genome_fasta(genome_fasta)

        sequences, skipped = extract_promoter_sequences(
            genes, tss_coords, genome_fasta,
            upstream_bp, downstream_bp, max_n_frac,
        )

        # Save FASTA for inspection and --from-fasta reruns
        with open(fasta_path, 'w') as f:
            for rat_id, seq in sequences.items():
                f.write(f">{rat_id}\n{seq}\n")
        logger.info(f"  FASTA saved: {fasta_path}")

        # Save skipped report
        skip_path = output_dir / 'promoter_skipped.tsv'
        with open(skip_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['rat_gene', 'reason'], delimiter='\t')
            writer.writeheader()
            writer.writerows(skipped)
        logger.info(f"  Skipped genes: {len(skipped):,} (see {skip_path.name})")

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 3: DNABert inference
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\nPhase 3: DNABert inference")
    logger.info("-" * 60)

    embeddings = build_promoter_embeddings(
        sequences, rat_to_human, dnabert_model, batch_size,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 4: Save embeddings + manifest
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\nPhase 4: Saving embeddings")
    logger.info("-" * 60)

    emb_path = output_dir / 'promoter_embeddings.pkl'
    with open(emb_path, 'wb') as f:
        pickle.dump(embeddings, f, protocol=4)
    logger.info(f"Embeddings saved: {emb_path}  ({len(embeddings):,} genes × {vector_size}d)")

    write_manifest(
        output_dir, config,
        n_genes_input=len(genes),
        n_sequences_extracted=len(sequences),
        n_skipped=len(skipped) if not args.from_fasta else 0,
        n_embeddings=len(embeddings),
        vector_size=vector_size,
        test_mode=args.test,
        elapsed=time.time() - t_start,
        dry_run=False,
        status='test_complete' if args.test else 'complete',
    )

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info(
        "STAGE 6 STEP 3 COMPLETE — Promoter Embedding"
        + (" (TEST MODE)" if args.test else "")
    )
    logger.info("=" * 60)
    logger.info(f"  Genes processed:    {len(genes):,}")
    logger.info(f"  Sequences extracted:{len(sequences):,}")
    logger.info(f"  Genes embedded:     {len(embeddings):,}")
    logger.info(f"  Elapsed:            {elapsed:.1f}s")
    logger.info(f"\n  Primary output:   {emb_path}")

    if args.test:
        logger.info("")
        logger.info("Test complete. Review outputs, then run full genome-wide:")
        logger.info("  python pipeline/06_prior_knowledge/build_promoter_embedding.py")


if __name__ == '__main__':
    main()