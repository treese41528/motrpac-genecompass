#!/usr/bin/env python3
"""
build_ortholog_mapping.py — Stage 3: Ortholog Mapping

Pipeline position:
    Stage 1: Data Harvesting & QC
    Stage 2: Gene Universe & Cell QC
      Step 1: build_gene_universe.py (scan → resolve → gene_universe.tsv)
      Step 2: preprocess_training_matrices.py (cell QC → raw-count h5ad)
      Step 3: prune_gene_universe.py (expression-based pruning → pruned_gene_universe.tsv)
    Stage 3: build_ortholog_mapping.py                          ← THIS SCRIPT
    Stage 4: Gene Medians (normalize → compute medians)
    Stage 5: Reference Assembly & Tokenization (median-divide → log2 → rank → top-2048)

Purpose:
    Map each rat gene in the pruned gene universe to a GeneCompass token ID
    via tiered ortholog resolution. Genes with cross-species orthologs inherit
    pre-trained token embeddings; genes without orthologs receive new token IDs
    that will be learned during fine-tuning.

Tier architecture (each rat gene assigned to first matching tier):
    T1   — Tri-species one2one: rat→human + rat→mouse, human↔mouse linked in GC
    T2a  — Human-rat one2one: rat→human in GC vocab
    T2b  — Mouse-rat one2one: rat→mouse in GC vocab
    T3a  — Human-rat beyond one2one: one2many / many2many to human in GC vocab
    T3b  — Mouse-rat beyond one2one: one2many / many2many to mouse in GC vocab
    T4   — New rat token: no qualifying ortholog, biotype ∈ {protein_coding, lncRNA, miRNA}

Key design decisions:
    - NO identity threshold — GeneCompass used none (empirically verified: min 1.2%).
      Ensembl's tree-based orthology inference IS the quality gate.
    - Human preferred over mouse (GC pre-training heavily human-weighted).
    - One2one preferred over one2many/many2many (unambiguous functional equivalence).
    - For one2many/many2many: disambiguate by highest % identity.
    - Token collisions tracked with expansion direction (rat-expanded vs human-expanded).
    - Confidence scoring (high/medium/low) as interpretive guardrail for Aims 2-3.

Inputs:
    pruned_gene_universe.tsv   — 22,213 rat genes from Stage 2
    rat_human_orthologs.tsv    — BioMart rat→human with orthology type + % identity
    rat_mouse_orthologs.tsv    — BioMart rat→mouse with orthology type + % identity
    rat_gene_info.tsv          — Rat gene metadata (symbol, biotype)
    human_mouse_tokens.pickle  — GeneCompass vocabulary (gene→token_id)
    gc_homologs.pickle         — GeneCompass homolog map (mouse_token→human_token)
    pipeline_config.yaml       — All configuration

Outputs:
    rat_token_mapping.tsv      — Master table with tier, confidence, collision diagnostics
    rat_tokens.pickle          — {rat_ensrnog: token_id}
    rat_to_human_mapping.pickle — {rat_ensrnog: human_ensg}
    rat_to_mouse_mapping.pickle — {rat_ensrnog: mouse_ensmusg}
    new_rat_tokens.txt         — Rat genes assigned new token IDs (Tier 4)
    tier_diagnostics.json      — Per-tier stats, identity distributions, biotype crosstabs
    collision_report.tsv       — Tokens shared by >1 rat gene
    mapping_statistics.json    — Full statistics
    stage3_manifest.json       — Config snapshot + input checksums

Usage:
    python pipeline/03_ortholog_mapping/build_ortholog_mapping.py
    python pipeline/03_ortholog_mapping/build_ortholog_mapping.py --dry-run
    python pipeline/03_ortholog_mapping/build_ortholog_mapping.py -v

Author: Tim Reese Lab / Claude
Date: February 2026
"""

import csv
import hashlib
import json
import logging
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Resolve project root from PIPELINE_ROOT env var or config default,
# then locate lib/ for shared utilities.
_PROJECT_ROOT = Path(os.environ.get('PIPELINE_ROOT', '.')).resolve()
sys.path.insert(0, str(_PROJECT_ROOT / 'lib'))
from gene_utils import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_pruned_gene_universe(path: Path) -> Dict[str, Dict]:
    """Load pruned_gene_universe.tsv from Stage 2, Step 3.

    Returns: {ensembl_id: {symbol, biotype, n_studies, ...}}
    """
    genes = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            eid = row['ensembl_id'].strip().upper()
            genes[eid] = {
                'symbol': row.get('symbol', ''),
                'biotype': row.get('biotype', ''),
                'n_studies': int(row.get('n_studies', 0)),
            }
    logger.info(f"Pruned gene universe: {len(genes):,} rat genes from {path.name}")
    return genes


def load_biomart_orthologs(path: Path, target_species: str) -> Dict[str, List[Dict]]:
    """Load BioMart ortholog table with auto-detected column names.

    Returns: {rat_ensrnog: [{target_gene, orth_type, perc_id, perc_id_target}, ...]}
    """
    candidates = defaultdict(list)
    n_records = 0

    with open(path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        headers = reader.fieldnames

        # Auto-detect columns by pattern matching
        col_map = _detect_ortholog_columns(headers, target_species)

        for row in reader:
            rat_id = row.get(col_map['rat_gene'], '').strip()
            if not rat_id:
                continue
            rat_id = rat_id.split('.')[0].upper()
            if not rat_id.startswith('ENSRNOG'):
                continue

            target_id = row.get(col_map['target_gene'], '').strip()
            if not target_id:
                continue
            target_id = target_id.split('.')[0].upper()

            orth_type = row.get(col_map.get('orth_type', ''), '').strip()
            perc_id_raw = row.get(col_map.get('perc_id', ''), '')
            perc_id_target_raw = row.get(col_map.get('perc_id_target', ''), '')

            try:
                perc_id = float(perc_id_raw) if perc_id_raw.strip() else None
            except (ValueError, AttributeError):
                perc_id = None

            try:
                perc_id_target = float(perc_id_target_raw) if perc_id_target_raw.strip() else None
            except (ValueError, AttributeError):
                perc_id_target = None

            candidates[rat_id].append({
                'target_gene': target_id,
                'orth_type': orth_type,
                'perc_id': perc_id,
                'perc_id_target': perc_id_target,
            })
            n_records += 1

    logger.info(f"BioMart {target_species} orthologs: {n_records:,} records, "
                f"{len(candidates):,} rat genes from {path.name}")
    return dict(candidates)


def _detect_ortholog_columns(headers: List[str], target_species: str) -> Dict[str, str]:
    """Auto-detect column names from BioMart ortholog file headers."""
    col_map = {}
    target_prefix = 'human' if target_species == 'human' else 'mouse'

    for h in headers:
        hl = h.strip().lower().replace(' ', '_')

        # Rat gene ID column
        if any(x in hl for x in ('gene_stable_id', 'gene_id', 'ensembl_gene_id')):
            if target_prefix not in hl and 'homolog' not in hl:
                col_map.setdefault('rat_gene', h)

        # Target gene ID
        if 'homolog' in hl and ('ensembl' in hl or 'gene' in hl) and 'name' not in hl:
            if 'associated' not in hl and 'type' not in hl and 'perc' not in hl:
                col_map.setdefault('target_gene', h)

        # Orthology type
        if 'orthology_type' in hl or 'homology_type' in hl:
            col_map.setdefault('orth_type', h)

        # Percent identity (rat → target)
        if 'perc' in hl and 'id' in hl:
            if 'target' in hl or 'r1' in hl:
                col_map.setdefault('perc_id_target', h)
            else:
                col_map.setdefault('perc_id', h)

    if 'rat_gene' not in col_map or 'target_gene' not in col_map:
        # Fallback: use positional mapping
        if len(headers) >= 2:
            col_map['rat_gene'] = headers[0]
            col_map['target_gene'] = headers[1]
            if len(headers) >= 4:
                col_map['orth_type'] = headers[3] if len(headers) > 3 else headers[2]
            if len(headers) >= 5:
                col_map['perc_id'] = headers[4] if len(headers) > 4 else ''
            logger.warning(f"Used positional column mapping for {target_species}: {col_map}")

    logger.debug(f"Column mapping for {target_species}: {col_map}")
    return col_map


def load_genecompass_vocab(path: Path) -> Dict[str, int]:
    """Load GeneCompass vocabulary: gene_id → token_id."""
    with open(path, 'rb') as f:
        vocab = pickle.load(f)
    n_human = sum(1 for k in vocab if isinstance(k, str) and k.startswith('ENSG'))
    n_mouse = sum(1 for k in vocab if isinstance(k, str) and k.startswith('ENSMUSG'))
    n_special = sum(1 for k in vocab if isinstance(k, str) and k.startswith('<'))
    logger.info(f"GeneCompass vocab: {len(vocab):,} tokens "
                f"({n_human:,} human, {n_mouse:,} mouse, {n_special} special)")
    return vocab


def load_genecompass_homologs(path: Path) -> Dict[int, int]:
    """Load GeneCompass homolog map: mouse_token_id → human_token_id."""
    with open(path, 'rb') as f:
        homologs = pickle.load(f)
    logger.info(f"GeneCompass homologs: {len(homologs):,} mouse→human token pairs")
    return homologs


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: INDEX BUILDING
# ═════════════════════════════════════════════════════════════════════════════

def build_gc_indices(
    gc_vocab: Dict[str, int],
    gc_homologs: Dict[int, int]
) -> Tuple[Dict[str, int], Dict[str, int], Set[Tuple[str, str]]]:
    """Build GeneCompass lookup indices.

    Returns:
        human_token:  {human_ensg: token_id}
        mouse_token:  {mouse_ensmusg: token_id}
        gc_linked:    {(mouse_gene, human_gene)} — pairs linked in GC homolog map
    """
    # Gene → token maps (gene IDs only, skip special tokens)
    human_token = {}
    mouse_token = {}
    for k, v in gc_vocab.items():
        if not isinstance(k, str):
            continue
        k_upper = k.upper() if isinstance(k, str) else k
        if k_upper.startswith('ENSG') and not k_upper.startswith('ENSMUSG'):
            human_token[k_upper] = v
        elif k_upper.startswith('ENSMUSG'):
            mouse_token[k_upper] = v

    # Reverse maps: token → gene
    token_to_human = {v: k for k, v in human_token.items()}
    token_to_mouse = {v: k for k, v in mouse_token.items()}

    # Build linked pairs set for fast Tier 1 lookup
    gc_linked = set()
    for m_tok, h_tok in gc_homologs.items():
        m_gene = token_to_mouse.get(m_tok)
        h_gene = token_to_human.get(h_tok)
        if m_gene and h_gene:
            gc_linked.add((m_gene, h_gene))

    logger.info(f"GC indices: {len(human_token):,} human, {len(mouse_token):,} mouse, "
                f"{len(gc_linked):,} linked pairs")
    return human_token, mouse_token, gc_linked


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: TIER ASSIGNMENT
# ═════════════════════════════════════════════════════════════════════════════

def assign_tiers(
    rat_genes: Dict[str, Dict],
    rat_human_candidates: Dict[str, List[Dict]],
    rat_mouse_candidates: Dict[str, List[Dict]],
    human_token: Dict[str, int],
    mouse_token: Dict[str, int],
    gc_linked: Set[Tuple[str, str]],
    config: Dict,
) -> List[Dict]:
    """Assign each rat gene to its highest-priority tier.

    Returns list of assignment dicts, one per rat gene.
    """
    orth_cfg = config.get('orthologs', {})
    one2one_types = set(orth_cfg.get('one2one_types', ['ortholog_one2one']))
    multi_types = set(orth_cfg.get('multi_types', ['ortholog_one2many', 'ortholog_many2many']))
    new_token_biotypes = set(orth_cfg.get('new_token_biotypes', ['protein_coding', 'lncRNA', 'miRNA']))

    # Compute next available token ID (after all existing GC tokens)
    all_token_ids = [v for v in list(human_token.values()) + list(mouse_token.values())
                     if isinstance(v, (int, float))]
    next_token_id = max(all_token_ids) + 1 if all_token_ids else 50000

    assignments = []
    tier_counts = Counter()

    for rat_gene in sorted(rat_genes.keys()):
        gene_info = rat_genes[rat_gene]
        biotype = gene_info.get('biotype', '')

        # Collect candidates annotated with GC vocab membership
        h_cands = _annotate_candidates(
            rat_human_candidates.get(rat_gene, []),
            human_token, one2one_types, multi_types
        )
        m_cands = _annotate_candidates(
            rat_mouse_candidates.get(rat_gene, []),
            mouse_token, one2one_types, multi_types
        )

        assignment = {
            'rat_gene': rat_gene,
            'rat_symbol': gene_info.get('symbol', ''),
            'biotype': biotype,
            'tier': None,
            'token_id': None,
            'human_ortholog': '',
            'mouse_ortholog': '',
            'orth_type_human': '',
            'orth_type_mouse': '',
            'perc_id_human': '',
            'perc_id_mouse': '',
        }

        # ── TIER 1: Tri-species one2one ──
        human_1to1 = [c for c in h_cands if c['is_one2one'] and c['in_gc_vocab']]
        mouse_1to1 = [c for c in m_cands if c['is_one2one'] and c['in_gc_vocab']]

        tier1_found = False
        for h in human_1to1:
            for m in mouse_1to1:
                if (m['target_gene'], h['target_gene']) in gc_linked:
                    assignment.update({
                        'tier': 'T1_tri_species',
                        'token_id': human_token[h['target_gene']],
                        'human_ortholog': h['target_gene'],
                        'mouse_ortholog': m['target_gene'],
                        'orth_type_human': h['orth_type'],
                        'orth_type_mouse': m['orth_type'],
                        'perc_id_human': h['perc_id'] if h['perc_id'] is not None else '',
                        'perc_id_mouse': m['perc_id'] if m['perc_id'] is not None else '',
                    })
                    tier1_found = True
                    break
            if tier1_found:
                break

        if tier1_found:
            tier_counts['T1_tri_species'] += 1
            assignments.append(assignment)
            continue

        # ── TIER 2a: Human-rat one2one ──
        if human_1to1:
            best = _pick_best_candidate(human_1to1)
            assignment.update({
                'tier': 'T2a_human_one2one',
                'token_id': human_token[best['target_gene']],
                'human_ortholog': best['target_gene'],
                'orth_type_human': best['orth_type'],
                'perc_id_human': best['perc_id'] if best['perc_id'] is not None else '',
            })
            # Record mouse ortholog if available (even if not in GC)
            if mouse_1to1:
                m_best = _pick_best_candidate(mouse_1to1)
                assignment['mouse_ortholog'] = m_best['target_gene']
                assignment['orth_type_mouse'] = m_best['orth_type']
                assignment['perc_id_mouse'] = m_best['perc_id'] if m_best['perc_id'] is not None else ''
            tier_counts['T2a_human_one2one'] += 1
            assignments.append(assignment)
            continue

        # ── TIER 2b: Mouse-rat one2one ──
        if mouse_1to1:
            best = _pick_best_candidate(mouse_1to1)
            assignment.update({
                'tier': 'T2b_mouse_one2one',
                'token_id': mouse_token[best['target_gene']],
                'mouse_ortholog': best['target_gene'],
                'orth_type_mouse': best['orth_type'],
                'perc_id_mouse': best['perc_id'] if best['perc_id'] is not None else '',
            })
            tier_counts['T2b_mouse_one2one'] += 1
            assignments.append(assignment)
            continue

        # ── TIER 3a: Human-rat beyond one2one ──
        human_any_gc = [c for c in h_cands if c['in_gc_vocab'] and not c['is_one2one']]
        if human_any_gc:
            best = _pick_best_candidate(human_any_gc)
            assignment.update({
                'tier': 'T3a_human_multi',
                'token_id': human_token[best['target_gene']],
                'human_ortholog': best['target_gene'],
                'orth_type_human': best['orth_type'],
                'perc_id_human': best['perc_id'] if best['perc_id'] is not None else '',
            })
            tier_counts['T3a_human_multi'] += 1
            assignments.append(assignment)
            continue

        # ── TIER 3b: Mouse-rat beyond one2one ──
        mouse_any_gc = [c for c in m_cands if c['in_gc_vocab'] and not c['is_one2one']]
        if mouse_any_gc:
            best = _pick_best_candidate(mouse_any_gc)
            assignment.update({
                'tier': 'T3b_mouse_multi',
                'token_id': mouse_token[best['target_gene']],
                'mouse_ortholog': best['target_gene'],
                'orth_type_mouse': best['orth_type'],
                'perc_id_mouse': best['perc_id'] if best['perc_id'] is not None else '',
            })
            tier_counts['T3b_mouse_multi'] += 1
            assignments.append(assignment)
            continue

        # ── TIER 4: New rat token ──
        if biotype.lower() in {b.lower() for b in new_token_biotypes}:
            assignment.update({
                'tier': 'T4_new_token',
                'token_id': next_token_id,
            })
            next_token_id += 1
            tier_counts['T4_new_token'] += 1
            assignments.append(assignment)
            continue

        # ── EXCLUDED ──
        assignment['tier'] = 'excluded'
        tier_counts['excluded'] += 1
        assignments.append(assignment)

    # Log tier distribution
    total = len(assignments)
    logger.info("Tier assignment complete:")
    for tier in ['T1_tri_species', 'T2a_human_one2one', 'T2b_mouse_one2one',
                 'T3a_human_multi', 'T3b_mouse_multi', 'T4_new_token', 'excluded']:
        cnt = tier_counts.get(tier, 0)
        pct = cnt / total * 100 if total else 0
        logger.info(f"  {tier:25s} {cnt:>6,}  ({pct:5.1f}%)")

    return assignments


def _annotate_candidates(
    candidates: List[Dict],
    token_map: Dict[str, int],
    one2one_types: Set[str],
    multi_types: Set[str],
) -> List[Dict]:
    """Annotate each candidate with GC vocab membership and orthology class."""
    annotated = []
    for c in candidates:
        target = c['target_gene'].upper()
        orth_type = c.get('orth_type', '')
        annotated.append({
            **c,
            'target_gene': target,
            'in_gc_vocab': target in token_map,
            'is_one2one': orth_type in one2one_types,
            'is_multi': orth_type in multi_types,
        })
    return annotated


def _pick_best_candidate(candidates: List[Dict]) -> Dict:
    """Pick the candidate with the highest % identity (disambiguation rule)."""
    return max(candidates, key=lambda c: c.get('perc_id') or 0)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: DIAGNOSTICS & CONFIDENCE SCORING
# ═════════════════════════════════════════════════════════════════════════════

def compute_diagnostics(assignments: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Compute token collision audit, expansion direction, and confidence scores.

    Returns:
        assignments:   updated with collision_count, expansion_direction, confidence
        diagnostics:   comprehensive diagnostics dict for JSON output
    """
    # ── Token collision audit ──
    token_to_rat_genes = defaultdict(list)
    for a in assignments:
        if a['token_id'] is not None and a['tier'] != 'T4_new_token':
            token_to_rat_genes[a['token_id']].append(a['rat_gene'])

    # Count collisions per token
    collision_counts = {}
    for token_id, rat_list in token_to_rat_genes.items():
        collision_counts[token_id] = len(rat_list)

    # ── Expansion direction ──
    # For each assignment, determine if the collision is rat-expanded
    # (multiple rat genes → 1 human/mouse token) or if it's from
    # the disambiguation side (1 rat gene picked from multiple targets)
    for a in assignments:
        tok = a['token_id']
        if tok is None:
            a['token_collision_count'] = 0
            a['expansion_direction'] = 'NA'
        elif a['tier'] == 'T4_new_token':
            a['token_collision_count'] = 1  # unique by construction
            a['expansion_direction'] = 'NA'
        else:
            cc = collision_counts.get(tok, 1)
            a['token_collision_count'] = cc
            if cc > 1:
                a['expansion_direction'] = 'rat_expanded'
            elif a['tier'] in ('T3a_human_multi', 'T3b_mouse_multi'):
                a['expansion_direction'] = 'human_expanded' if 'human' in a['tier'] else 'mouse_expanded'
            else:
                a['expansion_direction'] = 'one2one'

    # ── Confidence scoring ──
    for a in assignments:
        a['confidence'] = _score_confidence(a)

    # ── Build diagnostics summary ──
    diagnostics = _build_diagnostics_summary(assignments, collision_counts, token_to_rat_genes)

    return assignments, diagnostics


def _score_confidence(a: Dict) -> str:
    """Assign high/medium/low confidence per gene.

    Heuristic from implementation plan §6:
    - High:   Tier 1/2, one2one, %identity ≥ 70%, no collision, protein_coding
    - Medium: Tier 2-3, %identity 40-70%, collision ≤ 2, or non-coding with one2one
    - Low:    Tier 3 with collision > 2, %identity < 40%, non-coding with multi-type, or Tier 4
    """
    tier = a.get('tier', '')
    biotype = a.get('biotype', '').lower()
    collision = a.get('token_collision_count', 0)

    # Get best available percent identity
    perc_id = None
    for field in ('perc_id_human', 'perc_id_mouse'):
        val = a.get(field, '')
        if val != '' and val is not None:
            try:
                perc_id = float(val)
                break
            except (ValueError, TypeError):
                pass

    # Tier 4 or excluded → low
    if tier in ('T4_new_token', 'excluded'):
        return 'low'

    # Tier 1 or 2 with strong signal → high
    if tier in ('T1_tri_species', 'T2a_human_one2one', 'T2b_mouse_one2one'):
        if biotype == 'protein_coding' and collision <= 1:
            if perc_id is not None and perc_id >= 70:
                return 'high'
            elif perc_id is None:
                # Missing identity data but good tier → medium
                return 'medium'
        # Non-coding one2one → medium
        if biotype in ('lncrna', 'mirna') and collision <= 1:
            return 'medium'
        # Collision in one2one tier → medium (rat-expanded paralogs)
        if collision <= 2 and perc_id is not None and perc_id >= 40:
            return 'medium'

    # Tier 3 checks
    if tier in ('T3a_human_multi', 'T3b_mouse_multi'):
        if collision > 2:
            return 'low'
        if perc_id is not None and perc_id < 40:
            return 'low'
        if biotype in ('lncrna', 'mirna'):
            return 'low'
        if perc_id is not None and perc_id >= 40 and collision <= 2:
            return 'medium'

    # Default: medium for anything that didn't match high or low
    if tier in ('T1_tri_species', 'T2a_human_one2one', 'T2b_mouse_one2one'):
        return 'medium'
    return 'low'


def _build_diagnostics_summary(
    assignments: List[Dict],
    collision_counts: Dict[int, int],
    token_to_rat_genes: Dict[int, List[str]],
) -> Dict:
    """Build comprehensive diagnostics dict for tier_diagnostics.json."""
    tiers = ['T1_tri_species', 'T2a_human_one2one', 'T2b_mouse_one2one',
             'T3a_human_multi', 'T3b_mouse_multi', 'T4_new_token', 'excluded']

    # Per-tier statistics
    tier_stats = {}
    for tier in tiers:
        tier_genes = [a for a in assignments if a['tier'] == tier]
        if not tier_genes:
            tier_stats[tier] = {'count': 0}
            continue

        # Identity distributions
        human_ids = [float(a['perc_id_human']) for a in tier_genes
                     if a['perc_id_human'] not in ('', None)]
        mouse_ids = [float(a['perc_id_mouse']) for a in tier_genes
                     if a['perc_id_mouse'] not in ('', None)]

        # Biotype breakdown
        biotype_counts = Counter(a['biotype'] for a in tier_genes)

        # Confidence breakdown
        conf_counts = Counter(a.get('confidence', 'unknown') for a in tier_genes)

        stats = {
            'count': len(tier_genes),
            'biotype_breakdown': dict(biotype_counts),
            'confidence_breakdown': dict(conf_counts),
        }

        if human_ids:
            stats['human_identity'] = {
                'min': round(min(human_ids), 1),
                'p25': round(sorted(human_ids)[len(human_ids) // 4], 1),
                'median': round(sorted(human_ids)[len(human_ids) // 2], 1),
                'p75': round(sorted(human_ids)[3 * len(human_ids) // 4], 1),
                'max': round(max(human_ids), 1),
                'mean': round(sum(human_ids) / len(human_ids), 1),
                'n': len(human_ids),
            }
        if mouse_ids:
            stats['mouse_identity'] = {
                'min': round(min(mouse_ids), 1),
                'p25': round(sorted(mouse_ids)[len(mouse_ids) // 4], 1),
                'median': round(sorted(mouse_ids)[len(mouse_ids) // 2], 1),
                'p75': round(sorted(mouse_ids)[3 * len(mouse_ids) // 4], 1),
                'max': round(max(mouse_ids), 1),
                'mean': round(sum(mouse_ids) / len(mouse_ids), 1),
                'n': len(mouse_ids),
            }

        tier_stats[tier] = stats

    # Biotype × tier crosstab
    biotype_tier_crosstab = defaultdict(lambda: defaultdict(int))
    for a in assignments:
        biotype_tier_crosstab[a['biotype']][a['tier']] += 1

    # Collision summary
    collision_dist = Counter()
    for cc in collision_counts.values():
        if cc == 1:
            collision_dist['no_collision'] += 1
        elif cc == 2:
            collision_dist['collision_2'] += 1
        elif cc <= 5:
            collision_dist['collision_3_5'] += 1
        elif cc <= 10:
            collision_dist['collision_6_10'] += 1
        else:
            collision_dist['collision_10+'] += 1

    high_collision_tokens = {
        tok: genes for tok, genes in token_to_rat_genes.items()
        if len(genes) > 2
    }

    # Non-coding genes in Tiers 1-3 (flagged for review)
    flagged_noncoding = []
    for a in assignments:
        if a['biotype'] in ('lncRNA', 'miRNA') and a['tier'] in (
            'T1_tri_species', 'T2a_human_one2one', 'T2b_mouse_one2one',
            'T3a_human_multi', 'T3b_mouse_multi'
        ):
            flagged_noncoding.append({
                'rat_gene': a['rat_gene'],
                'symbol': a['rat_symbol'],
                'biotype': a['biotype'],
                'tier': a['tier'],
                'perc_id_human': a.get('perc_id_human', ''),
                'perc_id_mouse': a.get('perc_id_mouse', ''),
            })

    # Overall confidence distribution
    overall_conf = Counter(a.get('confidence', 'unknown') for a in assignments
                           if a['tier'] != 'excluded')

    diagnostics = {
        'tier_statistics': {k: v for k, v in tier_stats.items()},
        'biotype_tier_crosstab': {k: dict(v) for k, v in biotype_tier_crosstab.items()},
        'collision_summary': {
            'distribution': dict(collision_dist),
            'total_tokens_with_collision': sum(1 for cc in collision_counts.values() if cc > 1),
            'max_collision_count': max(collision_counts.values()) if collision_counts else 0,
            'high_collision_tokens_count': len(high_collision_tokens),
        },
        'flagged_noncoding_in_tiers_1_3': {
            'count': len(flagged_noncoding),
            'genes': flagged_noncoding[:50],  # Cap at 50 for readability
        },
        'overall_confidence': dict(overall_conf),
    }

    return diagnostics


def build_collision_report(assignments: List[Dict]) -> List[Dict]:
    """Build collision report: tokens shared by >1 rat gene."""
    token_to_assignments = defaultdict(list)
    for a in assignments:
        if a['token_id'] is not None and a['tier'] not in ('T4_new_token', 'excluded'):
            token_to_assignments[a['token_id']].append(a)

    rows = []
    for token_id, gene_assignments in sorted(token_to_assignments.items()):
        if len(gene_assignments) <= 1:
            continue
        for a in gene_assignments:
            rows.append({
                'token_id': token_id,
                'collision_count': len(gene_assignments),
                'rat_gene': a['rat_gene'],
                'rat_symbol': a['rat_symbol'],
                'biotype': a['biotype'],
                'tier': a['tier'],
                'human_ortholog': a.get('human_ortholog', ''),
                'mouse_ortholog': a.get('mouse_ortholog', ''),
                'perc_id_human': a.get('perc_id_human', ''),
                'perc_id_mouse': a.get('perc_id_mouse', ''),
                'expansion_direction': a.get('expansion_direction', ''),
            })

    return rows


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: OUTPUT
# ═════════════════════════════════════════════════════════════════════════════

def write_outputs(
    assignments: List[Dict],
    diagnostics: Dict,
    collision_rows: List[Dict],
    config: Dict,
    output_dir: Path,
    input_checksums: Dict[str, str],
):
    """Write all Stage 3 output files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. rat_token_mapping.tsv (master table) ──
    mapping_path = output_dir / 'rat_token_mapping.tsv'
    fieldnames = [
        'rat_gene', 'rat_symbol', 'biotype', 'tier', 'token_id',
        'human_ortholog', 'mouse_ortholog',
        'orth_type_human', 'orth_type_mouse',
        'perc_id_human', 'perc_id_mouse',
        'token_collision_count', 'expansion_direction', 'confidence',
    ]
    with open(mapping_path, 'w', newline='') as f:
        w = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for a in sorted(assignments, key=lambda x: x['rat_gene']):
            w.writerow(a)
    logger.info(f"rat_token_mapping.tsv: {len(assignments):,} genes")

    # ── 2. rat_tokens.pickle ──
    rat_tokens = {}
    for a in assignments:
        if a['token_id'] is not None and a['tier'] != 'excluded':
            rat_tokens[a['rat_gene']] = a['token_id']
    with open(output_dir / 'rat_tokens.pickle', 'wb') as f:
        pickle.dump(rat_tokens, f)
    logger.info(f"rat_tokens.pickle: {len(rat_tokens):,} mappings")

    # ── 3. rat_to_human_mapping.pickle ──
    rat_to_human = {}
    for a in assignments:
        if a.get('human_ortholog'):
            rat_to_human[a['rat_gene']] = a['human_ortholog']
    with open(output_dir / 'rat_to_human_mapping.pickle', 'wb') as f:
        pickle.dump(rat_to_human, f)
    logger.info(f"rat_to_human_mapping.pickle: {len(rat_to_human):,} mappings")

    # ── 4. rat_to_mouse_mapping.pickle ──
    rat_to_mouse = {}
    for a in assignments:
        if a.get('mouse_ortholog'):
            rat_to_mouse[a['rat_gene']] = a['mouse_ortholog']
    with open(output_dir / 'rat_to_mouse_mapping.pickle', 'wb') as f:
        pickle.dump(rat_to_mouse, f)
    logger.info(f"rat_to_mouse_mapping.pickle: {len(rat_to_mouse):,} mappings")

    # ── 5. new_rat_tokens.txt ──
    new_tokens = sorted(a['rat_gene'] for a in assignments if a['tier'] == 'T4_new_token')
    with open(output_dir / 'new_rat_tokens.txt', 'w') as f:
        f.write('\n'.join(new_tokens) + '\n')
    logger.info(f"new_rat_tokens.txt: {len(new_tokens):,} genes")

    # ── 6. tier_diagnostics.json ──
    with open(output_dir / 'tier_diagnostics.json', 'w') as f:
        json.dump(diagnostics, f, indent=2, default=str)
    logger.info(f"tier_diagnostics.json written")

    # ── 7. collision_report.tsv ──
    collision_path = output_dir / 'collision_report.tsv'
    if collision_rows:
        coll_fields = [
            'token_id', 'collision_count', 'rat_gene', 'rat_symbol', 'biotype',
            'tier', 'human_ortholog', 'mouse_ortholog',
            'perc_id_human', 'perc_id_mouse', 'expansion_direction',
        ]
        with open(collision_path, 'w', newline='') as f:
            w = csv.DictWriter(f, delimiter='\t', fieldnames=coll_fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(collision_rows)
        n_unique_tokens = len(set(r['token_id'] for r in collision_rows))
        logger.info(f"collision_report.tsv: {len(collision_rows):,} entries "
                     f"({n_unique_tokens} tokens with collisions)")
    else:
        with open(collision_path, 'w') as f:
            f.write("# No token collisions detected\n")
        logger.info("collision_report.tsv: no collisions")

    # ── 8. mapping_statistics.json ──
    tier_counts = Counter(a['tier'] for a in assignments)
    conf_counts = Counter(a.get('confidence', 'unknown') for a in assignments
                          if a['tier'] != 'excluded')
    biotype_counts = Counter(a['biotype'] for a in assignments if a['tier'] != 'excluded')

    total_mapped = sum(1 for a in assignments if a['tier'] != 'excluded')
    total_input = len(assignments)

    stats = {
        'timestamp': datetime.now().isoformat(),
        'total_input_genes': total_input,
        'total_mapped': total_mapped,
        'total_excluded': tier_counts.get('excluded', 0),
        'mapping_rate': round(total_mapped / total_input * 100, 2) if total_input else 0,
        'tier_distribution': {
            'T1_tri_species': tier_counts.get('T1_tri_species', 0),
            'T2a_human_one2one': tier_counts.get('T2a_human_one2one', 0),
            'T2b_mouse_one2one': tier_counts.get('T2b_mouse_one2one', 0),
            'T3a_human_multi': tier_counts.get('T3a_human_multi', 0),
            'T3b_mouse_multi': tier_counts.get('T3b_mouse_multi', 0),
            'T4_new_token': tier_counts.get('T4_new_token', 0),
            'excluded': tier_counts.get('excluded', 0),
        },
        'tier_percentages': {
            tier: round(cnt / total_input * 100, 2)
            for tier, cnt in tier_counts.items()
        },
        'confidence_distribution': dict(conf_counts),
        'biotype_distribution': dict(biotype_counts),
        'pre_trained_tokens_used': total_mapped - tier_counts.get('T4_new_token', 0),
        'new_tokens_created': tier_counts.get('T4_new_token', 0),
        'rat_to_human_mappings': len(rat_to_human),
        'rat_to_mouse_mappings': len(rat_to_mouse),
        'token_collisions': {
            'total_colliding_tokens': diagnostics['collision_summary']['total_tokens_with_collision'],
            'max_collision': diagnostics['collision_summary']['max_collision_count'],
        },
    }
    with open(output_dir / 'mapping_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"mapping_statistics.json written")

    # ── 9. stage3_manifest.json ──
    manifest = {
        'stage': 'stage3',
        'description': 'Ortholog Mapping — rat genes → GeneCompass tokens',
        'script': 'build_ortholog_mapping.py',
        'timestamp': datetime.now().isoformat(),
        'git_hash': _git_hash(),
        'config_snapshot': {
            'orthologs': config.get('orthologs', {}),
            'note': 'No identity threshold — Ensembl tree-based inference is the quality gate',
        },
        'input_checksums': input_checksums,
        'tier_counts': dict(tier_counts),
        'outputs': {
            'rat_token_mapping.tsv': len(assignments),
            'rat_tokens.pickle': len(rat_tokens),
            'rat_to_human_mapping.pickle': len(rat_to_human),
            'rat_to_mouse_mapping.pickle': len(rat_to_mouse),
            'new_rat_tokens.txt': len(new_tokens),
        },
        'downstream': {
            'stage4': 'Gene Medians — rat_to_human_mapping.pickle determines human vs rat medians',
            'stage5': 'Tokenization — rat_tokens.pickle is the tokenizer for rat cells',
            'aim2_grn': 'confidence column → GRN edges with low-confidence genes flagged',
            'aim3_translation': 'confidence restricts cross-species translation claims',
        },
    }
    with open(output_dir / 'stage3_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"stage3_manifest.json written")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def _md5(path: Path) -> str:
    """Compute MD5 checksum of a file."""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _git_hash() -> str:
    """Get current git commit hash."""
    try:
        import subprocess
        r = subprocess.run(['git', 'rev-parse', 'HEAD'],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def build_ortholog_mapping(config: dict, dry_run: bool = False):
    """Main pipeline: load → tier assignment → diagnostics → output."""
    t_start = time.time()

    paths = config['paths']
    bm = config['biomart']
    orth_cfg = config.get('orthologs', {})

    # Resolve all input paths
    gene_universe_path = resolve_path(config, paths['gene_universe_dir']) / 'pruned_gene_universe.tsv'
    rat_human_path = resolve_path(config, bm['rat_human_orthologs'])
    rat_mouse_path = resolve_path(config, bm['rat_mouse_orthologs'])
    gc_vocab_path = resolve_path(config, paths['genecompass_tokens'])
    gc_homolog_path = resolve_path(config, paths['genecompass_homologs'])
    output_dir = resolve_path(config, paths['ortholog_dir'])

    # Validate inputs
    required_files = {
        'pruned_gene_universe.tsv': gene_universe_path,
        'rat_human_orthologs.tsv': rat_human_path,
        'rat_mouse_orthologs.tsv': rat_mouse_path,
        'genecompass_tokens': gc_vocab_path,
        'genecompass_homologs': gc_homolog_path,
    }

    logger.info("=" * 60)
    logger.info("STAGE 3: ORTHOLOG MAPPING")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config: identity_threshold=None (GC used none)")
    logger.info(f"Config: disambiguation={orth_cfg.get('disambiguation', 'highest_identity')}")

    missing = []
    for name, fpath in required_files.items():
        exists = fpath.exists()
        status = 'OK' if exists else 'MISSING'
        logger.info(f"  {name}: {fpath} [{status}]")
        if not exists:
            missing.append(name)

    if missing:
        logger.error(f"Missing input files: {missing}")
        logger.error("Ensure Stages 1-2 have been run and BioMart references are downloaded.")
        sys.exit(1)

    if dry_run:
        logger.info("DRY RUN — all inputs validated, exiting")
        return

    # ── Load data ──
    logger.info("")
    logger.info("Loading data...")
    rat_genes = load_pruned_gene_universe(gene_universe_path)
    rat_human_cands = load_biomart_orthologs(rat_human_path, 'human')
    rat_mouse_cands = load_biomart_orthologs(rat_mouse_path, 'mouse')
    gc_vocab = load_genecompass_vocab(gc_vocab_path)
    gc_homologs = load_genecompass_homologs(gc_homolog_path)

    # ── Build indices ──
    logger.info("")
    logger.info("Building GeneCompass indices...")
    human_token, mouse_token, gc_linked = build_gc_indices(gc_vocab, gc_homologs)

    # ── Tier assignment ──
    logger.info("")
    logger.info("Assigning tiers...")
    assignments = assign_tiers(
        rat_genes, rat_human_cands, rat_mouse_cands,
        human_token, mouse_token, gc_linked, config,
    )

    # ── Diagnostics ──
    logger.info("")
    logger.info("Computing diagnostics...")
    assignments, diagnostics = compute_diagnostics(assignments)
    collision_rows = build_collision_report(assignments)

    # ── Compute input checksums ──
    input_checksums = {}
    for name, fpath in required_files.items():
        input_checksums[name] = _md5(fpath)

    # ── Write outputs ──
    logger.info("")
    logger.info("Writing outputs...")
    write_outputs(assignments, diagnostics, collision_rows, config, output_dir, input_checksums)

    # ── Summary ──
    elapsed = time.time() - t_start
    tier_counts = Counter(a['tier'] for a in assignments)
    conf_counts = Counter(a.get('confidence', '') for a in assignments if a['tier'] != 'excluded')
    total = len(assignments)
    total_mapped = sum(1 for a in assignments if a['tier'] != 'excluded')

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 3 SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Input genes:           {total:,}")
    logger.info(f"  Mapped to tokens:      {total_mapped:,} ({total_mapped/total*100:.1f}%)")
    logger.info(f"  Excluded:              {tier_counts.get('excluded', 0):,}")
    logger.info("")
    logger.info("  Tier distribution:")
    for tier in ['T1_tri_species', 'T2a_human_one2one', 'T2b_mouse_one2one',
                 'T3a_human_multi', 'T3b_mouse_multi', 'T4_new_token']:
        cnt = tier_counts.get(tier, 0)
        pct = cnt / total * 100 if total else 0
        logger.info(f"    {tier:25s} {cnt:>6,}  ({pct:5.1f}%)")
    logger.info("")
    logger.info("  Confidence distribution:")
    for level in ['high', 'medium', 'low']:
        cnt = conf_counts.get(level, 0)
        pct = cnt / total_mapped * 100 if total_mapped else 0
        logger.info(f"    {level:10s} {cnt:>6,}  ({pct:5.1f}%)")
    logger.info("")
    logger.info(f"  Token collisions:      {diagnostics['collision_summary']['total_tokens_with_collision']}")
    logger.info(f"  Max collision count:   {diagnostics['collision_summary']['max_collision_count']}")
    n_flagged = diagnostics['flagged_noncoding_in_tiers_1_3']['count']
    logger.info(f"  Non-coding in T1-T3:   {n_flagged} (flagged for review)")
    logger.info(f"  Elapsed:               {elapsed:.1f}s")
    logger.info(f"  Output:                {output_dir}")
    logger.info("")
    logger.info("  Downstream handoff:")
    logger.info("    Stage 4 → rat_to_human_mapping.pickle (human vs rat medians)")
    logger.info("    Stage 5 → rat_tokens.pickle (tokenizer for rat cells)")
    logger.info("    Aim 2  → confidence column (GRN edge quality flags)")
    logger.info("    Aim 3  → confidence column (cross-species translation guardrails)")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 3: Build rat ortholog mappings for GeneCompass fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tier Architecture:
  T1   Tri-species one2one — rat→human + rat→mouse, human↔mouse linked in GC
  T2a  Human-rat one2one   — rat→human in GC vocab
  T2b  Mouse-rat one2one   — rat→mouse in GC vocab
  T3a  Human-rat multi     — one2many/many2many to human in GC vocab
  T3b  Mouse-rat multi     — one2many/many2many to mouse in GC vocab
  T4   New rat token       — no ortholog, biotype = protein_coding/lncRNA/miRNA

Key: NO identity threshold. GeneCompass used none. Ensembl tree-based
     orthology inference IS the quality gate.

Examples:
  python pipeline/03_ortholog_mapping/build_ortholog_mapping.py
  python pipeline/03_ortholog_mapping/build_ortholog_mapping.py --dry-run
  python pipeline/03_ortholog_mapping/build_ortholog_mapping.py -v
        """
    )
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate config and inputs, then exit')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        config = load_config()
    except FileNotFoundError:
        logger.error("pipeline_config.yaml not found. Set PIPELINE_ROOT or run from project root.")
        sys.exit(1)

    for section in ('orthologs', 'biomart', 'paths'):
        if section not in config:
            logger.error(f"Config missing '{section}' section")
            sys.exit(1)

    build_ortholog_mapping(config, dry_run=args.dry_run)


if __name__ == '__main__':
    main()