#!/usr/bin/env python
# coding: utf-8
"""
build_rat_tf_list.py -- Module A.5: the rat transcription-factor list to perturb (Aim 2b/3b).

Produces deconvolution/reference/rat_tf_ensrnog.tsv: rel-113 ENSRNOG rat TFs (in the GeneCompass vocab),
optionally restricted to per-tissue expressed genes. This is the "which genes to delete" list for the
GRN (Module B): perturb each TF -> targets -> differential trained-vs-control network.

SOURCE (one small external fetch; see --tf-symbols):
  Provide a plain-text list of TF gene symbols (one per line), from either
    - Lambert et al. 2018 "The Human Transcription Factors":
      http://humantfs.ccbr.utoronto.ca/download/v_1.01/TF_names_v_1.01.txt   (1,639 human TFs), OR
    - AnimalTFDB (Rattus norvegicus TF list), symbols column.
  We map by SYMBOL against rat_token_mapping.tsv `rat_symbol` (case-insensitive) AND against the human
  ortholog symbol, so either a human or a rat symbol list works. (Rat symbols are largely shared with
  human; the ortholog fallback catches the rest.) ENSRNOG only, intersected with the in-vocab tokens.

Usage:
  python finetune/genecompass/build_rat_tf_list.py --tf-symbols data/references/tf/TF_names_v_1.01.txt \
      [--expressed-dir data/deconvolution/results_novis/motrpac] [--out deconvolution/reference/rat_tf_ensrnog.tsv]
"""
import argparse
import csv
import glob
import os
import pickle
from pathlib import Path

_ROOT = Path(os.environ.get("PIPELINE_ROOT", ".")).resolve()
MAP = _ROOT / "data" / "training" / "ortholog_mappings" / "rat_token_mapping.tsv"
TOK = _ROOT / "data" / "training" / "ortholog_mappings" / "rat_tokens.pickle"


def expressed_genes(expressed_dir):
    """union of per-tissue pred_z gene lists -> set of ENSRNOG expressed somewhere (if a dir is given)."""
    if not expressed_dir:
        return None
    genes = set()
    for gf in glob.glob(os.path.join(expressed_dir, "**", "genes.txt"), recursive=True):
        genes |= {l.strip() for l in open(gf) if l.strip().startswith("ENSRNOG")}
    return genes or None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf-symbols", required=True, help="text file of TF gene symbols, one per line")
    ap.add_argument("--expressed-dir", default=None, help="restrict to genes with a pred_z/genes.txt (optional)")
    ap.add_argument("--out", default=str(_ROOT / "deconvolution" / "reference" / "rat_tf_ensrnog.tsv"))
    args = ap.parse_args()

    tf_syms = {l.strip().upper() for l in open(args.tf_symbols) if l.strip() and not l.startswith(("#", ">"))}
    tok = pickle.load(open(TOK, "rb"))                    # ENSRNOG -> token idx (in-vocab set)
    expr = expressed_genes(args.expressed_dir)

    # human ENSG -> human symbol is not on disk; we match on rat_symbol and, where the list is human,
    # rat_symbol == human symbol for most orthologs. (A human-ENSG symbol map could be added later.)
    rows = []
    with open(MAP) as f:
        rd = csv.DictReader(f, delimiter="\t")
        for r in rd:
            ens = r["rat_gene"]
            sym = (r.get("rat_symbol") or "").upper()
            if ens not in tok:                            # must be in the GeneCompass vocab to perturb
                continue
            if sym and sym in tf_syms:
                if expr is not None and ens not in expr:
                    continue
                rows.append((ens, r.get("rat_symbol", ""), r.get("tier", ""), r.get("token_id", ""),
                             r.get("human_ortholog", ""), r.get("confidence", "")))

    rows.sort(key=lambda x: x[1] or x[0])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("ensrnog\tsymbol\ttier\ttoken_id\thuman_ortholog\tconfidence\n")
        for row in rows:
            f.write("\t".join(row) + "\n")
    print(f"wrote {len(rows)} rat TFs (in-vocab{', expressed' if expr else ''}) -> {args.out}")
    if len(rows) < 500:
        print("  NB: <500 TFs -- check the symbol source matched rat_symbol casing/synonyms "
              "(a human-ENSG->symbol map would recover ortholog-only-named TFs).")


if __name__ == "__main__":
    main()
