# E.3 on the 16-tissue default (pre-registered 2026-11-04, before any SPLEEN or OVARY atlas cell was embedded)

**Why.** E.3 (`ATLAS_PROJECTION_PREREG.md`, result `ATLAS_PROJECTION_RESULTS.md`: PASS, 0.404 vs null 0.084) was
run on the refs4 (14-tissue) transferred cells. The default is now the 16-tissue clean rebuild. This reruns the
same test on the default's transferred cells and adds the two new tissues, SPLEEN and OVARY.

**Rat side.** The transferred pseudo-cells of the default, `data/deconvolution/genecompass_input_v2_refs4all_rebuild_human/<tissue>/`
(the scorer's default `--human-root`). Tissues with a Tabula Sapiens counterpart: the 11 of the first run plus
SPLEEN and OVARY (13). CORTEX, HIPPOC and HYPOTH stay out of scope (no brain in Tabula Sapiens).

**Human side.**
- The nine Tabula Sapiens tissues of the first run are reused as prepared and embedded
  (`data/deconvolution/atlas_tabula_sapiens/<Tissue>/`): same collection revision, same v2 checkpoint, same target
  sum (6,400 in every `prep_summary.json`, equal to the default's `pseudocell_target_sum`).
- New: Tabula Sapiens **Spleen** and **Ovary** from the same CELLxGENE collection
  (e5f58829-1a66-40b5-a624-9046778e74f5, revision 2026-06-11; dataset files 5498457f-6a18-4766-b9a3-a2c8c4662703
  and 6ad78e84-6715-458d-95b4-69e8509cd785), md5s added to `data/external/tabula_sapiens/MANIFEST.txt`. Prepared
  and embedded by exactly the first run's rules (10x only, one pseudobulk per donor × cell type with ≥ 20 cells,
  the transfer's tokenization, ≥ 200 genes, `embed_cells.py --species 0`).
- `project_to_atlas.py` gains the tissue pairs spleen → Spleen and ovary → Ovary; nothing else in it changes.

**Correspondence map.** The 129 existing rows of `translation/atlas_correspondence.tsv` are not changed. Rows
for the rat SPLEEN (14) and OVARY (18) cell types are written from names alone (the rat labels and the Tabula
Sapiens `cell_type` values with at least one donor pseudobulk, listed from the h5ad metadata), and committed
before any SPLEEN or OVARY atlas cell is embedded. The same rules as the first map apply: a rat type maps to
every Tabula Sapiens type it names or is a subtype of; empty when there is no counterpart.

**Test and reading.** Unchanged: within-tissue centring, nearest centroid by cosine, macro top-1 accuracy over
all scored rat types, label-shuffle null (10,000 permutations), PASS if p < 0.05 and accuracy ≥ 2 × null mean,
WEAK if p < 0.05 but less, FAIL if p ≥ 0.05. Uncentred embeddings reported without a verdict. Output:
`data/deconvolution/atlas_tabula_sapiens/score_refs4all_rebuild/` (the first run's `score/` is kept).

**Also reported, without a verdict.**
- The nine current hotspots, each with its accuracy and most common human type.
- SPLEEN and OVARY per type.
- The verdict over the 11 first-run tissues alone. Reproduction check: their per-type accuracies against the
  first run's (`translation/atlas_record/`). Their transferred cells are byte-identical to refs4's
  (`REFS4ALL_REBUILD_RESULTS.md`), so the accuracies should be equal. The null and p change slightly, because
  the permutation draws now span 13 tissues.

No other centring, distance, pseudobulk size or map revision will be tried after the results are seen.
