# Scheme B: the per-gene medians that rat counts are divided by

**Question this serves.** GeneCompass reads a cell as a ranked list of genes. The rank comes from each gene's
count divided by a fixed per-gene median, so that a gene is ranked by how far it is above its usual level and
not by its absolute abundance. For rat cells to be read by a model pretrained on human and mouse cells, the
divisor of each rat gene has to sit on the scale the pretrained model saw. "Scheme B" is the rule that
chooses that divisor. It is used wherever rat counts are tokenized: the training corpus (Stage 5) and the
pseudo-cells (Stage 9). The human re-expression of Stage 12 uses GeneCompass's human medians directly.

Until 2026-11-14 the rule was defined only in the message of commit `ab5843a` (2026-10-06) and in a scratch
file outside the repository (audit finding 2.4). This page is the definition. Every number in the first three
sections comes from a file named beside it; the table is
`docs/audit/remediation/record/step6/scheme_b_medians.tsv`, made by
`analysis/audit_remediation/step6_scheme_b_record.py`.

## The rule

For a rat gene *g* with a Stage 4 rat median:

| Case | Divisor of *g* |
|---|---|
| The human ortholog of *g* (Stage 3, `rat_to_human_mapping.pickle`) has a median in GeneCompass's `human_gene_median_after_filter.pickle` | that human median |
| Otherwise | the Stage 4 rat median of *g*, divided by **F** |

**F** is one number for the whole build: the median, over the genes that have both a rat median and a human
median, of (rat median / human median). It is needed because the Stage 4 rat medians are computed on counts
scaled to 10,000 per cell, while GeneCompass's medians are on raw counts, so the rat medians sit about twice
as high. Without F, genes on a rat median entered each cell at about half the value of their neighbours.

A gene with no Stage 4 rat median has no divisor and is never tokenized.

- **Builder.** `pipeline/05_tokenization/build_hybrid_medians.py --fallback-rescale auto` (`auto` is the
  default). `--fallback-rescale none` leaves the rat medians unscaled, which is what April's model used
  ("scheme A").
- **Outputs.** `hybrid_gene_medians.pickle` and `hybrid_median_provenance.json` in the folder of the config
  key `paths.median_dir`; the config key `paths.hybrid_medians` points at the pickle.
- **Normalisation.** Raw counts are divided by these medians with no per-cell depth normalisation, then
  log2(1 + x), then the 2,048 highest values are kept (`pipeline/05_tokenization/tokenize_corpus.py`).

## The v2 build

| Quantity | Value | Where it is recorded |
|---|---|---|
| Fallback mode | `auto` | `hybrid_median_provenance.json`, `fallback_rescale.mode` |
| F | 2.052656 (2.0527) | same file, `fallback_rescale.scale`; recomputed from the inputs: 2.052656 |
| Genes with both medians (F is their median ratio) | 14,766 | same file, `fallback_rescale.n_shared_genes`; recomputed: 14,766 |
| Genes in the dictionary | 21,222 | `hybrid_gene_medians.pickle` |
| ... on a human median | 14,766 (69.58%) | `hybrid_median_provenance.json`, `coverage.human_median` |
| ... on a rat median divided by F | 6,456 (30.42%) | same file, `coverage.rat_median` |
| ... on the constant fallback 1.0 | 0 | same file, `coverage.fallback` |
| Genes of the Stage 3 universe with no median, never tokenized | 991 of 22,213 | `rat_token_mapping.tsv` against the pickle |
| ... of which new tokens (T3a, T3b, T4) | 613 | same |
| Written | 2026-10-09 02:41 UTC | `hybrid_median_provenance.json`, `generated_at` |

The files are in the folder of `paths.median_dir` (`data/training/gene_medians_v2/` under the default
configuration; md5 of the pickle `3bcd501ef49209636d74da6b8360d4f9`).

By ortholog tier (`record/step6/scheme_b_source_by_tier.tsv`):

| Tier | Genes | Human median | Rat median / F | No median |
|---|---|---|---|---|
| T1 tri-species | 13,178 | 13,116 | 21 | 41 |
| T2a human one-to-one | 705 | 560 | 100 | 45 |
| T2b mouse one-to-one | 3,613 | 0 | 3,321 | 292 |
| T3a human multi | 1,351 | 1,090 | 40 | 221 |
| T3b mouse multi | 892 | 0 | 681 | 211 |
| T4 new token | 2,474 | 0 | 2,293 | 181 |
| All | 22,213 | 14,766 | 6,456 | 991 |

So "rat median" is not "tier 4": it is every gene without a GeneCompass human median, including 161 genes
that do have a human ortholog (T1, T2a, T3a) whose human gene has no median in GeneCompass's file.

## Check that the file on disk follows the rule

`step6_scheme_b_record.py` rebuilds the dictionary from the three inputs with the builder's own functions
and compares it, gene by gene, with the pickle on disk (2026-11-14):

- 14,766 of 14,766 human-median genes hold exactly the human median of their ortholog;
- 6,456 of 6,456 rat-median genes hold exactly their rat median divided by the recomputed F;
- 21,222 of 21,222 values and 21,222 of 21,222 recorded sources are identical to the rebuilt ones.

The reproduction of 2026-10-31 reports that it rebuilt the same file byte for byte from a Stage 4 rerun
(`docs/repro_stages1_7/REPRODUCTION.md`, Stage 4 row). That was not repeated here.

## Why B, and what the evidence was

Four schemes were compared on 2026-10-06 against GeneCompass's own human and mouse corpora. **The evidence
for this section is not in the repository.** It is a scratch table, log and script in an uncommitted local
folder (`tmp/data_audit/median_check/`: `median_check_summary.tsv`, `check_11893348.out`,
`check_medians.py`). The figures below were read from those files on 2026-11-14. The files were not copied
into the repository and the comparison was not rerun, so this table cannot be regenerated from a clone.

| Set | Median stored value | Per-gene median, p50 |
|---|---|---|
| GeneCompass human corpus | 1.015 | 1.379 |
| GeneCompass mouse corpus | 1.007 | 1.371 |
| A: human median, else unscaled rat median (April) | 0.886 | 1.113 |
| **B: human median, else rat median / F** | **0.946** | **1.160** |
| C: rat median / F for every gene | 0.950 | 1.144 |
| D: rat median for every gene, unscaled | 0.501 | 0.623 |

The log gives 20,370 rat cells from 582 QC files. Under A, genes on a rat median had a median value of 0.499
in the same cells where genes on a human median had 0.936; under B the two are 0.963 and 0.946. The stated
reason for B over C is in the message of commit `ab5843a`: B keeps the human-median anchor, so a rat cell and
its human re-expression share the divisor of every gene that has one.

## Limits

- **The decision evidence is uncommitted** (previous section). Committing the three scratch files, or
  rerunning the comparison from a committed script, would close that.
- **The comparison was run on the March medians**, computed before the corpus clean-up (F = 2.242, 14,812
  and 6,567 genes, by the scratch log). It was not repeated on v2's medians (F = 2.0527).
- **The v2 build's command line and commit are not recorded.** The medians folder holds
  `hybrid_median_provenance.json` (which records mode `auto` and F) and no
  `provenance_build_hybrid_medians.json`.
- **Paralogs share a divisor.** The audit counts 15,234 rat genes mapping to 14,476 human genes, so some rat
  paralogs are divided by the same human median (audit Part 2, Stage 4 deviation 5; not recounted here).
- **F is one global constant** for 6,456 genes. It absorbs the difference between "counts scaled to 10,000"
  and "raw counts" on average, not gene by gene.
- **991 genes have no median**, 613 of them new tokens that the model therefore never sees (audit finding 2.6).
