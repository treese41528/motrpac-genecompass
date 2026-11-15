#!/usr/bin/env python3
"""step6_scheme_b_record.py -- Step 6 (records, tests, hygiene), audit finding 2.4.

"Scheme B" is the rule that gives each rat gene the median its counts are divided by before ranking (Stage 5
and Stage 9). It was defined only in a commit message (ab5843a) and a scratch file. This script reads the
files of the default configuration (paths.median_dir, paths.ortholog_dir, paths.genecompass_medians) and
checks, gene by gene, that the hybrid dictionary on disk is what the rule says:

    human median of the rat gene's human ortholog, where GeneCompass has one;
    otherwise the gene's own Stage 4 rat median divided by F,
    F = median over the genes that have both of (rat median / human median).

It recomputes F with the builder's own function and compares it with the value the builder recorded.
Read-only.

  python analysis/audit_remediation/step6_scheme_b_record.py
Out (docs/audit/remediation/record/step6/): scheme_b_medians.tsv (every figure, with the file it comes from)
and scheme_b_source_by_tier.tsv (genes by ortholog tier and median source).
"""
import csv
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "pipeline" / "05_tokenization"))
from gene_utils import load_config, resolve_path                               # noqa: E402
import build_hybrid_medians as bhm                                              # noqa: E402

OUT = ROOT / "docs/audit/remediation/record/step6"


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def rel(p):
    p = Path(p)
    return str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p)


def main():
    cfg = load_config(); paths = cfg["paths"]
    median_dir = resolve_path(cfg, paths["median_dir"])
    orth_dir = resolve_path(cfg, paths["ortholog_dir"])
    gc_medians = resolve_path(cfg, paths["genecompass_medians"])
    prov_f = median_dir / "hybrid_median_provenance.json"
    hybrid_f = median_dir / "hybrid_gene_medians.pickle"
    rat_f = median_dir / "rat_gene_medians.pickle"

    rat = bhm.load_rat_medians(median_dir)
    r2h = bhm.load_rat_to_human(orth_dir)
    human = bhm.load_human_medians(gc_medians)
    hybrid = pickle.load(open(hybrid_f, "rb"))
    prov = json.load(open(prov_f))
    src = prov["per_gene_source"]
    F, n_shared = bhm.measure_fallback_scale(rat, r2h, human)

    # the rule, applied gene by gene to the inputs, against the dictionary on disk
    rebuilt, rebuilt_src = bhm.build_hybrid_medians(rat, r2h, human, fallback_scale=F)
    same_keys = set(rebuilt) == set(hybrid)
    n_equal = sum(1 for g in hybrid if g in rebuilt and rebuilt[g] == hybrid[g])
    n_src_equal = sum(1 for g in src if rebuilt_src.get(g) == src[g])
    n_h_exact = sum(1 for g, s in src.items() if s == "human" and hybrid[g] == human[r2h[g]])
    n_r_exact = sum(1 for g, s in src.items() if s == "rat" and hybrid[g] == rat[g] / F)

    tiers, universe = {}, 0
    with open(orth_dir / "rat_token_mapping.tsv") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            universe += 1
            tiers[r["rat_gene"].split(".")[0].upper()] = r["tier"]
    by = Counter((tiers.get(g, "not in rat_token_mapping.tsv"), s) for g, s in src.items())
    no_median = Counter(t for g, t in tiers.items() if g not in hybrid)

    cov, fb = prov["coverage"], prov["fallback_rescale"]
    P, T = rel(prov_f), rel(orth_dir / "rat_token_mapping.tsv")
    rows = [
        ("builder", "pipeline/05_tokenization/build_hybrid_medians.py --fallback-rescale auto (the default)", "the script"),
        ("fallback mode recorded for this build", fb["mode"], f"{P}: fallback_rescale.mode"),
        ("scale F recorded (rounded to 6 decimals by the builder)", fb["scale"], f"{P}: fallback_rescale.scale"),
        ("scale F recomputed here from the three inputs", f"{F:.6f}", "measure_fallback_scale() on the inputs below"),
        ("genes with both a rat and a human median (F is their median ratio), recorded", fb["n_shared_genes"],
         f"{P}: fallback_rescale.n_shared_genes"),
        ("the same, recomputed here", n_shared, "measure_fallback_scale()"),
        ("genes in the hybrid dictionary", len(hybrid), rel(hybrid_f)),
        ("genes recorded with a human median", cov["human_median"], f"{P}: coverage.human_median"),
        ("genes recorded with a rescaled rat median", cov["rat_median"], f"{P}: coverage.rat_median"),
        ("genes recorded with the constant fallback 1.0", cov["fallback"], f"{P}: coverage.fallback"),
        ("genes whose value equals the human median of their ortholog, exactly", f"{n_h_exact} of {cov['human_median']}",
         f"{rel(hybrid_f)} against {rel(gc_medians)} through rat_to_human_mapping.pickle"),
        ("genes whose value equals their rat median divided by the recomputed F, exactly",
         f"{n_r_exact} of {cov['rat_median']}", f"{rel(hybrid_f)} against {rel(rat_f)}"),
        ("rule rebuilt from the inputs: same genes as the dictionary on disk", same_keys, "build_hybrid_medians()"),
        ("rule rebuilt from the inputs: genes with an identical value", f"{n_equal} of {len(hybrid)}", "build_hybrid_medians()"),
        ("rule rebuilt from the inputs: genes with the recorded source", f"{n_src_equal} of {len(src)}",
         f"{P}: per_gene_source"),
        ("genes in the Stage 3 universe", universe, T),
        ("universe genes with no median (never tokenized)", sum(no_median.values()), f"{T} against {rel(hybrid_f)}"),
        ("of those, genes with a new token (T3a, T3b or T4)",
         sum(v for k, v in no_median.items() if k.startswith(("T3", "T4"))), T),
        ("when the dictionary was written (UTC)", prov["generated_at"], f"{P}: generated_at"),
        ("command line and git commit of this build", "not recorded: the folder holds no provenance_build_hybrid_medians.json"
         if not (median_dir / "provenance_build_hybrid_medians.json").exists() else "recorded",
         rel(median_dir)),
        ("md5 hybrid_gene_medians.pickle", md5(hybrid_f), rel(hybrid_f)),
        ("md5 rat_gene_medians.pickle", md5(rat_f), rel(rat_f)),
        ("md5 rat_to_human_mapping.pickle", md5(orth_dir / "rat_to_human_mapping.pickle"),
         rel(orth_dir / "rat_to_human_mapping.pickle")),
        ("md5 human_gene_median_after_filter.pickle", md5(gc_medians), rel(gc_medians)),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "scheme_b_medians.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["quantity", "value", "source"]); w.writerows(rows)
    order = ["T1_tri_species", "T2a_human_one2one", "T2b_mouse_one2one", "T3a_human_multi", "T3b_mouse_multi",
             "T4_new_token"]
    with open(OUT / "scheme_b_source_by_tier.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["tier", "genes_in_universe", "human_median", "rat_median_div_F", "no_median"])
        n_tier = Counter(tiers.values())
        for t in order + sorted(set(n_tier) - set(order)):
            w.writerow([t, n_tier[t], by[(t, "human")], by[(t, "rat")], no_median[t]])
        w.writerow(["all", universe, cov["human_median"], cov["rat_median"], sum(no_median.values())])
    for q, v, s in rows:
        print(f"{q:<86} {v}")
    print(open(OUT / "scheme_b_source_by_tier.tsv").read())


if __name__ == "__main__":
    main()
