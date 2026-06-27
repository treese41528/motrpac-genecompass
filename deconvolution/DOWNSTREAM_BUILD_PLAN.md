# Downstream GeneCompass modules -- detailed build plan

_v1, 2026-06-22. Covers the proposal's UNBUILT downstream-GeneCompass tasks:
Aim 2b (GRN), Aim 2c (dose), Aim 3a/3b/3c (cross-species). Grounded in a read-only
inventory of `vendor/GeneCompass/` + the project + the interface of
`perturb_delete_chipseq.py` and `finetune/genecompass/embed_cells.py`._

Legend: **[exists]** built / on disk, **[build]** to create. Effort: S (<1 day),
M (1-3 days), L (1-2 weeks). Paths are relative to the repo root.

---

## 0. Principles (carry into every module)

- **P1 -- One keystone engine.** The rat in-silico-perturbation engine is built
  once (Module A) and reused by Aim 2b GRN (B) and Aim 3b conserved regulators (F).
  Do not fork it per aim.
- **P2 -- Abundant cell types only** for perturbation/GRN. Rare-in-tissue types sit
  at the reference prior (no exercise signal); restrict to `mean_fraction >= 0.01`
  (the DE power floor).
- **P3 -- Dose is ORDINAL** (week 1/2/4/8), never trained-vs-control binary. Control
  for sex. Frame results RELATIVE/differential (activity/composition confound; read
  `Z` against `theta`).
- **P4 -- Pre-register validation** like the DE: each module commits its acceptance
  criteria + positive controls BEFORE looking at outputs (see A.6).
- **P5 -- Single venv, no conda.** Vendored modules ship conda `.yaml` envs; install
  their real deps into `motrpac-env` (or a pinned container). Never `conda activate`;
  flag dep conflicts before installing.
- **P6 -- Confidence-gate cross-species claims** via the `rat_token_mapping.tsv`
  confidence column; report enrichment/relative effects, not large absolute magnitudes.
- **P7 -- SLURM only for GPU/multi-minute work** (dummy `--gres=gpu:1`, standby <= 4h);
  seconds-scale sanity checks on the login node are fine.

## Shared inputs (already on disk)

| Input | Path |
|---|---|
| Fine-tuned rat model | `deconvolution.genecompass_model_dir` -> `data/models/rat_genecompass_finetuned/models/rat_phase2_mixed_species` (`config.json` + `.../models/pytorch_model.bin`; vocab 55275x768, 4 knowledge tensors) |
| Tokenized pseudo-cells | `data/deconvolution/genecompass_input/<tissue>/dataset` (cols: input_ids, values, length, species=2, cell_id, sample, cell_type, tissue) |
| CLS embeddings | `.../<tissue>/embeddings/cell_embeddings.npy` |
| Rat vocab / token dicts | `data/training/ortholog_mappings/{rat_tokens.pickle, rat_human_mouse_tokens.pickle (4,717 ENSRNOG keys), rat_token_mapping.tsv (human_ortholog+confidence), rat_to_human_mapping.pickle (15,234 / 94.5%)}` |
| Hotspots | `.../genecompass_input/corroboration_merged.tsv` (q_sup_trained<0.05 -> 22 blocks) |
| DE outputs (target gate) | `.../pseudobulk_de/{de_summary.tsv, de__*.tsv}` |
| Loader pattern to copy | `finetune/genecompass/embed_cells.py:87-128` |

---

## Milestone 0 -- close-out (in flight; not new modules)

Finish Stage 10: run `compare_posctrl.py` on the DE outputs, verify
`posctrl_summary.md` against the frozen miss-ladder, commit the new DE/Stage-10
files on a branch, write the DE results into `AIM2_DECONV_RESULTS.md` section 5.1.
Effort S. (Tasks #4/#5/#6/#7.) **Everything below depends on the DE being final.**

---

## Module A -- rat in-silico perturbation engine  `[build]`  KEYSTONE  (effort M-L, GPU)

Aim 3b engine; also powers Aim 2b. It is `embed_cells.py`'s loader + the vendored
perturber's index-manipulation + cosine-shift scoring.

**A.1 New file:** `finetune/genecompass/perturb_cells.py` (beside `embed_cells.py`).

**A.2 Reuse + exact adaptations** of `vendor/GeneCompass/genecompass/perturb_delete_chipseq.py`:

Reuse as-is: `delete_index` (l.53), `overexpress_index` (l.64), `make_perturbation_batch`
(l.83; stacks input_ids+values into `[:,:,0]`/`[:,:,1]`), `perturb_emb_by_index` (l.47),
`make_comparison_batch` (l.137), `cos_sim_shift` (l.270), `pad_tensor_list` (l.280),
`class InSilicoPerturber` (l.299) control flow / `quant_cos_sims` (l.204).

Replace / fix (the rat-readiness edits):
- **(a) Model load** -- do NOT use the vendor `load_model` / `utils.load_prior_embedding`
  (human/mouse). Load exactly as `embed_cells.py:87-128`: knowledges from state-dict keys
  `bert.embeddings.{promoter,co_exp,gene_family,peca_grn}_knowledge`; `homologous_gene_human2mouse`
  from `vendor/.../prior_knowledge/homologous_hm_token.pickle`; patch
  `model.bert.cls_embedding = nn.Embedding(3, hidden_size)`; `load_state_dict(strict=False)`;
  `model.eval().to(device)`.
- **(b) Species** -- every forward sets `species=2` (rat). Vendor hardcodes 0 at l.38, l.237
  and `df['species']=0` (~l.538). Override ALL to 2.
- **(c) Device** -- de-hardcode `'cuda:1'` (l.40, 187, 236, 249, 597, ...) to a `--device` arg
  (default `cuda`), same as `embed_cells.py`.
- **(d) Token dict** -- pass the dict whose KEYS are gene ENSEMBL ids and VALUES are token
  indices, so `genes_to_perturb=[ENSRNOG...]` maps correctly. **VERIFY which pickle this is**
  (`rat_human_mouse_tokens.pickle` has the 4,717 ENSRNOG keys; confirm id->index, not
  index->id) via a 5-gene round-trip (symbol -> ENSRNOG through `pruned_gene_universe.tsv`
  -> token index) before wiring.
- **(e) Value scale** -- `overexpress_index` multiplies the value channel (vendor ~l.65,
  factor 10000). Our `values` are the tokenizer's ranked/normalized scale. Confirm the
  overexpress semantics on OUR scale with a 1-cell dry run before trusting magnitudes;
  prefer `delete` first (unambiguous).

**A.3 Inputs:** `--model-dir` (resolve via `genecompass_model_dir` + `resolve_model_dir`),
`--dataset .../genecompass_input/<tissue>/dataset`, `--genes` (rat TF/target ENSRNOG list or
`all`), `--perturb {delete,overexpress}`, `--cell-types` (default = abundant hotspot cell
types), `--device`, `--batch-size`.

**A.4 Outputs:** `.../genecompass_input/<tissue>/perturb/perturb__<celltype>__<perturb>.pkl`
(cosine-shift keyed by (perturbed_gene, affected_gene) and/or per-cell CLS shift) +
`perturb_manifest.json` (checkpoint hash, species=2, device, gene-list hash, vendor commit).

**A.5 Sub-build -- rat TF/target list** `[build]` (effort S):
`finetune/genecompass/build_rat_tf_list.py` -> `deconvolution/reference/rat_tf_ensrnog.tsv`.
Source: rat AnimalTFDB / human TF set mapped via `rat_to_human` reverse or RGD symbols ->
ENSRNOG (`pruned_gene_universe.tsv`, the same bridge the pre-reg used). Intersect with
per-tissue expressed genes (`pred_z/genes.txt`). ENSRNOG only (match the 4,717 rat tokens).
Analogous to vendor `data/insilico_perturbation/ids_in_chip_remain.pickle`.

**A.6 Pre-registered validation** (commit before running; mirror `POSCTRL_PREREG`):
- Positive control: perturbing a known muscle master-TF (Mef2c) in SKMVL/SKMGN skeletal-muscle
  cells produces a larger CLS shift than a random matched-expression gene; predicted top targets
  enrich for known Mef2 targets. Deleting a hematopoietic TF in blood monocytes likewise.
- Sanity: deleting a gene NOT expressed in the cell type -> ~zero shift.
- Self-consistency: shift magnitude correlates with the gene's value/rank.
- Freeze these (genes + expected direction + miss-reading ladder: not-expressed -> low-shift
  expected; abundant+expressed+flat -> real negative) into `deconvolution/PERTURB_PREREG.md`
  before the first real run.

**A.7 Risks:** vendor code is rough (trailing-space bug `'input_ids '` ~l.96; `self.ensembl_id`
last-loop-var ~l.424; absolute `/home/ict` paths in `__main__`). Treat as template; unit-test
`make_perturbation_batch` on a 2-cell toy before scaling. GeneCompass has no cross-cell
attention -> perturbation is strictly per-cell.

---

## Module B -- model-driven GRN (Aim 2b)  `[build]`  (effort L, no GPU; post-proc)

**B.1 New file:** `deconvolution/build_grn.py` (pure python).

**B.2 Method** (the docs' PREFERRED route; data-driven DeepSEM rejected, n~50 too thin):
for each abundant hotspot cell type, run Module A `delete` over candidate regulators
(`rat_tf_ensrnog.tsv`); a predicted edge regulator->target = target among the top-k most
shifted (lowest-cosine) genes on that regulator's deletion; assemble a directed adjacency;
build SEPARATE trained vs control networks (subset pseudo-cells by week) and report the
DIFFERENTIAL network (edges gained/lost with training).

**B.3 Inputs:** Module A pickles; `de__*.tsv` hotspot rankings to seed regulator/target sets;
`estimated_fractions.csv` (confound). **B.4 Outputs:**
`.../<tissue>/grn/grn__<celltype>__{trained,control,delta}.tsv` + differential-hub summary.

**B.5 Optional refinement (defer):** feed gene-embedding cosine-similarity into bundled
`vendor/.../downstream_tasks/grn_inference/` (DeepSEM-master) -- needs a conda env + rat
ground-truth edges (ChIP scarce for rat). Document as optional validation.

**B.6 Validation:** recover known TF->target edges (Mef2 in muscle; SPI1/CEBPA in monocytes).
Frame edges relative/differential. **B.7 Depends on:** Module A; abundant cell types only.

---

## Module C -- CPA dose model (Aim 2c)  `[build]`  (effort L, GPU)

**C.1 New file:** `deconvolution/run_cpa_dose.py` wrapping
`vendor/GeneCompass/downstream_tasks/drug_dose_response/cpa/`.
**C.2 Method:** week (ordinal 0/1/2/4/8) = dose covariate, sex = discrete covariate; train CPA
on hotspot pseudo-cells (Z and/or 768-d embeddings) to model + interpolate the latent dose
response; compare CPA trajectory vs the DE ordinal slope (`slope_week`).
**C.3 Inputs:** hotspot Z and/or embeddings; week+sex from `motrpac_sample_pheno.tsv`; CPA code.
**C.4 Outputs:** `.../<tissue>/cpa/cpa__<celltype>_{model.pt,dose_response.tsv,latent.npy}` +
DE-vs-CPA concordance report.
**C.5 Risks:** n~50/cell type is thin for an autoencoder -> lean on 5x replication, pool dose
levels, and distrust interpolation the DE slope does not corroborate. Install CPA deps into
the venv (P5). **C.6 Depends on:** DE; secondary priority.

---

## Module D -- cross-species human-genetics validation (Aim 3c)  `[build]`  (effort L)

Most tractable big deliverable; the hard hop (ortholog map) is DONE.

**D.0 Sub-build (start NOW; no GPU, no DE dep; effort M):** `translation/stage_human_genetics.py`
-> download + version + manifest the PUBLIC inputs: GTEx v8 eQTL; the ~114 GWAS set Vetr uses;
S-PrediXcan models (MASHR/JTI) + covariances (PredictDB); an Open Targets release. Record which
GTEx tissue maps to each of our 10 tissues. Add a `config/pipeline_config.yaml` block +
checksums (mirror the Stage-3 manifest).

**D.1 New file:** `translation/validate_human.py`.
- Step 1 (control): replicate Vetr's BULK result -- map Vetr's bulk DE genes -> human via
  `rat_to_human_mapping.pickle`, run GTEx/GWAS/S-PrediXcan/Open Targets, confirm we regenerate
  ~5,523 trait-tissue-gene triplets. Proves the port before any novelty.
- Step 2 (novelty): swap in OUR per-cell-type DE (`de__*.tsv`) -> human ortholog -> same steps
  -> CELL-TYPE-resolved trait-tissue-gene triplets.

**D.2 Reuse:** port `github.com/NikVetr/MoTrPAC_Complex_Traits` (Zenodo 10211801); Vetr gene
lists already wired in `compare_posctrl.py` (Tier B).
**D.3 Inputs:** per-cell-type DE; `rat_to_human_mapping.pickle` + confidence tier; staged
resources (D.0). **Outputs:** `data/results/cross_species/celltype_trait_triplets.tsv` + tables.
**D.4 Cautions:** effects small-but-real -> report enrichment, confidence-gate (P6), read Z vs
theta, handle many-to-many (T3) orthologs explicitly.
**D.5 Depends on:** DE final (D.1 step 2); D.0 can precede; NOT gated on the model.

---

## Module E -- cross-species TRANSFER of the rat exercise response into human embedding space (Aim 3a)  `[E.1/E.2 BUILT 2026-06-24, validating]`  (effort M, GPU)

THE MAIN PURPOSE of the cross-species work. MoTrPAC's invasive, multi-tissue, time-course exercise
study CANNOT be run on humans (ethics + feasibility) -- so the rat is the experimental proxy and
GeneCompass is the TRANSFER vehicle: re-express the measured rat exercise data in the HUMAN embedding
space, then analyze it there. This is one-directional cross-species TRANSFER (rat data -> human space
-> analyze), NOT alignment of two measured datasets. A human atlas is OPTIONAL (validation/interpretation
backdrop) -- NOT a binding gap. The output is a COUNTERFACTUAL human representation of the rat response
(an inference), not a human measurement.

> **BUILT 2026-06-24** -- `translation/transfer_to_human.py` (E.1) + `pipeline/run_stage12.py`
> (orchestrator) + `translation/compare_transfer.py` (E.2) + `slurm/analysis/run_stage12.slurm`.
> Tokenization parity verified against the actual rat path before writing a line (5-agent map +
> 4-lens adversarial review): target_sum **6500** (not the corpus 10000 -- all 10 rat runs use 6500),
> top-2048, log2(1+x/median), species=0. The "rat checkpoint can't embed human tokens" worry was
> DISPROVEN: rat tokens reuse the GeneCompass ID space, so 13,883/15,234 ortholog-mapped rat genes
> already carry their human ortholog's token ID; human ENSG token IDs are 2..23114, inside the 55,275
> checkpoint vocab; only species==1 (mouse) triggers homolog remapping, so human (0) uses identity
> lookup. Review caught + fixed a real value-channel parity bug (normalize_total must use the FULL rat
> library incl. dropped T4 genes, not the mapped-only library) -> liver value median now 0.847 (rat
> 0.872 / corpus 0.869) instead of an inflated 0.974; and a latent embed_cells row-shuffle on
> `--n-cells < n` (now an order-preserving prefix; fixes the rat path too). Per-tissue ortholog
> coverage ~82% of genes / ~81% of count-mass (liver 13,134->10,720 eligible human genes); the ~18%
> T4 rat-specific mass is dropped from the sequence but kept in the normalize denominator. Output
> layout = `genecompass_input_human/<tissue>/{dataset,embeddings,pseudocells.h5ad symlink}` so the
> existing detection scripts run unchanged with `--gc-root genecompass_input_human`.

**E.1 -- transfer (`translation/transfer_to_human.py`, effort M, GPU):** re-embed every rat exercise
pseudo-cell AS human. GeneCompass is cross-species (vocab has 23,113 human ENSG tokens + a species token):
  rat ENSRNOG expression
    --ortholog map (`rat_to_human_mapping.pickle`, 15,234 / 94.5%)--> human ENSG profile
    --tokenize (human ENSG tokens + human/hybrid gene medians, SAME target-sum 6500 + top-N as rat)-->
    --embed (same fine-tuned GeneCompass checkpoint, `species=0` human)--> HUMAN-space embeddings of the
      rat exercise cells.
  Reuses the `embed_cells.py` loader + the (done) ortholog map; NO human dataset needed. Output:
  `genecompass_input/<tissue>/embeddings_human/cell_embeddings.npy`. **Tokenization PARITY is make-or-break**
  (human ENSG tokens, human gene medians, same target-sum/top-N, species=0) -- a parity bug silently
  corrupts the transfer; ~5% non-mappable rat genes are dropped (logged).

**E.2 -- analyze in human space (the deliverable):** re-run the exercise-signal detection (gate /
supervised probe / Augur) and the within-cell-type DE axis ON the transferred embeddings -- does the
trained-vs-control / ordinal-dose axis SURVIVE the transfer, per cell type? That is the core test that the
rat response is human-transferable. Output: per-cell-type human-space exercise-axis report; compare to the
rat-space detection (signal persists / weakens / shifts). Combine with Module D (genetics) for the human
disease interpretation of the transferred signatures (3a transfers the representation; 3c gives the clinic).

**E.3 -- validation + interpretation (effort S/M):**
- **Transfer-validity sanity check FIRST:** a transferred rat monocyte must land near real human monocytes.
  Validate against a human reference atlas (Tabula Sapiens -- blood + skeletal muscle, CELLxGENE h5ad) used
  ONLY as an interpretive backdrop; recover known cell-type identity before trusting the transferred axis.
- **Confidence-gate** by ortholog tier (`rat_token_mapping.tsv`). Cross-species exercise conservation is
  imperfect (some exerkines go OPPOSITE rat<->human: IL15/BDNF/TGFB2), so the transfer is EMPIRICALLY tested,
  never assumed.

**E.4 Inputs:** rat pseudo-cells/embeddings (exist) + `rat_to_human_mapping.pickle` (done) + the fine-tuned
model + human/hybrid gene medians; OPTIONAL Tabula Sapiens for E.3. **Depends on:** the ortholog map (done)
+ `embed_cells.py` (exists) -> the core transfer (E.1/E.2) is buildable NOW, **no human-data dependency**.

---

## Module F -- conserved regulators (Aim 3c integration)  `[build]`  (effort S)

**F.1 New file:** `translation/conserved_regulators.py` -- join Module A/B perturbation+GRN hits
with Module D human-genetics triplets to flag TF/regulator networks conserved rat->human AND
disease-relevant, confidence-gated. Mostly join + ranking + visualization.
**F.2 Depends on:** Modules A/B and D.

---

## Module G -- viewer v2 (per-cell-type local PCA)  `[build]`  (effort M, lowest value)

**G.1** Extend `deconvolution/build_umap_viewer.py`: Focus tab projects each (tissue x cell-type)
onto its OWN local PCA of the 768-d embeddings, instead of one pooled PCA.
**G.2 Inputs:** `cell_embeddings.npy` + pheno; current `umap_viewer_template.html`. Pure
visualization; no model compute.

---

## Orchestration

Extend the numbered-stage convention (whole-experiment drivers; mirror `run_stage9/10`):
- **Stage 10 [exists]** -- Aim-2 analysis: DE -> positive-control comparison.
- **Stage 11 [build]** -- model-driven downstream: step 1 `perturb_cells.py` (A) -> step 2
  `build_grn.py` (B) -> step 3 `run_cpa_dose.py` (C).
- **Stage 12 [E.1/E.2 BUILT 2026-06-24]** -- cross-species: `run_stage12.py` drives step 1
  `transfer_to_human.py` (E.1 project+tokenize, CPU) -> step 2 `embed_cells.py --species 0` (GPU) ->
  step 3 `subspace_probe.py --gc-root <human-root>` (primary E.2 detector) -> step 4
  `compare_transfer.py` (E.2 deliverable: does the axis survive?). STILL [build]: D.0
  `stage_human_genetics.py`, D `validate_human.py`, F `conserved_regulators.py`, and folding the
  pheno/Augur corroboration + E.3 human-atlas backdrop into the driver.

Each: subprocess driver + `--from` + `--dry-run` + config paths + step validation, exactly like
`pipeline/run_stage10.py`. Viewer v2 (G) stays a standalone script.

## Dependency graph / critical path

```
  DE final (M0) --> A (perturb engine) --> B (GRN) --> F (conserved reg)
                \-> C (CPA dose)                       ^
                \-> D.1 (human validation) ------------/
  D.0 (stage genetics data)             : start NOW, independent.
  A.5 (rat TF list)                     : start NOW, independent.
  E.1/E.2 (transfer rat->human + analyze): start NOW (ortholog map + embed_cells exist; GPU). No human data.
  E.3 (human-atlas backdrop)            : OPTIONAL -- transfer-validity sanity check + interpretation only.
```

**Critical path:** M0 -> A -> B/F. Highest-leverage parallel pre-work with no DE dependency:
**D.0** (genetics data), **A.5** (TF list), **E.1/E.2** (transfer the rat exercise data into human space and
test whether the dose axis survives -- no human-data dependency, the core 3a deliverable).

## Recommended order

1. M0 close-out (DE + comparison + commit + write-up).  *(in flight)*
2. In parallel now: A.5 rat TF list; D.0 stage human genetics; **E.1/E.2 transfer the rat exercise data into human embedding space + test whether the dose axis survives** (no human-data dependency).
3. A perturbation engine (keystone) -- validate on Mef2c/muscle positive control.
4. B model-driven GRN (trained vs control), abundant hotspots only.
5. D human-genetics validation (replicate Vetr bulk, then cell-type novelty).
6. C CPA dose (secondary), E.3 human-atlas backdrop (OPTIONAL, validates the transfer), F conserved regulators.
7. G viewer v2 (last; visualization polish).

## Out of scope (do not build)

GEARS, drug-CPA (compounds), DeepCE/L1000 gene-expression-profiling: all need drug/CRISPR
perturb-seq TRAINING data we do not have for rat/MoTrPAC (exercise is not a molecular
perturbation). The CPA *exercise-dose* repurposing (Module C) is the only CPA use that fits.
Data-driven DeepSEM GRN is rejected (n~50 too thin) -- hence the model-driven route (A/B).
