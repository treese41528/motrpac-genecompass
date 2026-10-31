# E.3: anchoring the rat→human transfer to Tabula Sapiens (pre-registered 2026-10-31)

Registered before any Tabula Sapiens cell was embedded. The correspondence map
(`translation/atlas_correspondence.tsv`) was written from the cell-type names alone (rat reference labels and
the Tabula Sapiens `cell_type` values with at least one donor pseudobulk), with no embedding computed.

**Question.** The transfer re-expresses each rat pseudo-cell as a human profile and embeds it with the human
species token. Its checks so far reuse rat labels, so they show the transfer keeps the rat structure, not that
a transferred rat monocyte lands where real human monocytes are. Does it?

**Human side.** Tabula Sapiens (CELLxGENE collection e5f58829-1a66-40b5-a624-9046778e74f5, revision of
2026-06-11), the tissues with a MoTrPAC counterpart: Blood, Lung, Muscle (SKMGN and SKMVL), Heart, Liver, Kidney,
Fat (WATSC and BAT), Small_Intestine, Testis. There is no brain in Tabula Sapiens, so CORTEX, HIPPOC and HYPOTH
are out of scope. 10x cells only (Smart-seq2 counts are on another scale). One pseudobulk per (donor, cell type)
with at least 20 cells, summed raw counts, so each human point is a pseudobulk like the rat pseudo-cells. Each
is tokenized by the transfer's own math (normalize_total to the rat run's target sum, 6,400, over all genes;
human tokens and medians; top 2,048; at least 200 genes) and embedded with the v2 checkpoint, species 0
(`translation/project_to_atlas.py prep`, `embed_cells.py --species 0`).

**Rat side.** The transferred pseudo-cells of the default (refs4) run,
`data/deconvolution/genecompass_input_v2_refs4_human/<tissue>/`.

**Test (`project_to_atlas.py score`).** Within each tissue, both sets are centred on their own mean embedding
(the two sources differ in technology and in how a pseudobulk is formed, so a global offset is expected and is
not the question). Each Tabula Sapiens cell type's centroid is the mean of its centred donor pseudobulks. Each
transferred rat pseudo-cell is assigned the cell type with the highest cosine. A rat cell type's accuracy is the
fraction of its pseudo-cells assigned to one of its mapped human types; rat types with no counterpart in the
map are not scored.

- **Primary statistic:** macro mean accuracy over all scored rat cell types, all tissues.
- **Null:** within each tissue, shuffle which centroid carries which name (one draw per tissue per
  permutation), 10,000 permutations; p = (1 + #null ≥ observed) / (1 + 10,000).
- **Reading:** PASS if p < 0.05 and the accuracy is at least twice the null mean; WEAK if p < 0.05 but less
  than twice; FAIL if p ≥ 0.05. PASS means the transfer is anchored: transferred rat cells land nearest their
  real human counterparts. FAIL means this atlas does not anchor it, and the manuscript's limitation (v)
  stands as written.
- **Reported without a verdict:** per tissue and per rat cell type (the 11 hotspots among them), the most
  common assigned human type, and the same test on uncentred embeddings.

Interpretive, not a gate: a FAIL weakens cell-type claims made in human space; it does not undo the rat-space
results. No other centring, distance, pseudobulk size or map revision will be tried after the results are seen.
