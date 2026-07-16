# MoTrPAC rat BayesPrism reference-selection plan (2026-07-15)

Response to Geyu's `manuscript/wip/claude_note.md`: pick, per MoTrPAC PASS1B tissue, the best **rat**
single-cell/nucleus BayesPrism reference, filtering on **organism**, **geo_title**, and **control arms**;
prefer one study across tissues where clean; add TESTES; repair CORTEX; find references for the missing
tissues. Produced from a 19-tissue fan-out (one researcher per tissue, online-accession verification,
consolidation), with every load-bearing `sample_id`/GSM/arm claim re-checked against the inventory.

Bulk source to match: **MoTrPAC PASS1B = Fischer 344 (F344), adult ~6 mo, sedentary, bulk RNA-seq, 19 tissues.**

> **STATUS 2026-07-15:** all 13 buildable references BUILT (`data/deconvolution/references_v3/`) + VALIDATED at
> 3 levels incl. an adversarial GEO-record check (**all 13 CONFIRMED**); full deconvolution SUBMITTED. The
> reproducible rebuild system, the validation guarantees, and the full verification record are in
> **`deconvolution/DECONV_REBUILD_RUNBOOK.md`**.

---

## 1. The keystone fix — SHIPPED: `select_samples()` now filters on organism, title, and arm

`deconvolution/build_reference.py::select_samples()` used to filter on **accession + tissue + in_corpus only**.
That is the single bug that let a ~85%-non-rat CORTEX reference (GSE303115 spans 6 species) and a 52%-rat-mouse
chimera SKMVL reference reach production. Four filters were added (composed in precedence, all before the
existing reference-QC gate; `--sample-ids` stays the most-specific final pin):

| Flag | Default | Purpose |
|---|---|---|
| `--organism` | `Rattus norvegicus` | geo_organism gate. `--organism any` restores old behavior. **The keystone.** |
| `--title-include` | — | regex over `geo_title\|geo_source_name\|geo_cell_type`; selects arms living only in the title (Ma-2020 `-Y`; GSE248413 `Y`). |
| `--title-exclude` | — | drop by title regex (e.g. `visium\|spatial`). |
| `--no-dedup-gsm` | dedup ON | collapse duplicate GSMs (the `_RAW` double-ingest) to one sample_id per library. |

Precedence: `base → organism → conditions → sex → title-include → title-exclude → sample-ids → GSM-dedup → QC gate`.

**Verified behavior** (exercised against the real inventory — all pass):
- CORTEX `GSE303115 --tissue cortex` (defaults) → **2 rat GSMs** (`sample18, sample20`); the 6-species rows drop out; `--organism any` → 22 rows (old bug).
- HEART `GSE280111 --tissue "left ventricle"` (defaults) → **19** (dedup 38→19).
- Ma-2020 `--title-include '-Y($|\b|_)'`: muscle → `sample16,sample30`; WAT → `sample10,sample35`; BM → `sample20,sample9`.
- HYPOTH `GSE248413 --title-include '(^|[^A-Za-z])Y([^A-Za-z]|$)'` → **`sample0` only** (excludes the old arm, which `condition_resolved="no treatment"` could NOT distinguish).
- KIDNEY `--conditions "No treatment"` → 4 (deployed behavior preserved).

> These new defaults (rat-only, dedup-on) also apply to `make_pseudobulk.py`, `make_purity_sweep.py`,
> `compute_true_z.py`, `build_lung_pooled.py` (all import `load_study`). They are correct improvements
> everywhere; they change those outputs only on the next re-run.

---

## 2. Final reference decision — all 19 tissues

Status: **READY** = deployed & correct · **NEEDS-BUILD** = rebuild from corpus/staged data · **NEEDS-ONLINE-DATA** =
download/ingest first · **BLOCKED** = no rat reference exists anywhere.

Confidence and shared-source are folded into "Why chosen / replaced"; the full shared-source map is §3.

| Tissue | Study | Arm / sample_ids | Assay·strain | Status | Why chosen / replaced |
|---|---|---|---|---|---|
| KIDNEY | [GSE240658] | `--conditions "No treatment"` (4,6,8,15) | snRNA · SD | **READY** | **Kept (deployed).** snRNA captures the ~80% proximal-tubule parenchyma (holdout 0.9991); chosen over Ma scRNA, which under-captures PT (§3). |
| LIVER | [GSE220075] | sham snRNA (1,3,7,10); QC drops Visium | snRNA · Lewis+DA | **READY** | **Kept (deployed).** The only cross-validated tissue (DWLS ρ=0.943); snRNA captures hepatocytes. Chosen over Ma scRNA. |
| CORTEX | [GSE303115] | rat-only default → 2 GSMs; `--label-scheme brain`, **inner** join | snRNA · n/s | NEEDS-BUILD | **Replaced the invalid deployed `union_merged`** (was ~85% non-rat — built with no organism filter). Rat-only inner-join rebuild; only native rat cortex snRNA. |
| HEART | [GSE280111] | `--tissue "left ventricle"` → 19 dedup GSMs | snRNA · Wistar | NEEDS-BUILD | **Chosen** as the only clean rat LV atlas (CM r=0.995); dedup fixes the 38→19 double-count. Replaced [GSE155699] (SHR model, no cardiomyocytes). |
| VENACV | [GSE280111] | `--tissue "pulmonary veins"` → 8 dedup (venous proxy) | snRNA · Wistar | NEEDS-BUILD | **Best venous proxy** — no rat vena cava scRNA exists anywhere; shares HEART's atlas (a vein, unlike aorta). Differential-only. |
| WATSC | [GSE137869] | `--title-include '-Y…'` → 10,35 (young arm) | scRNA · SD | NEEDS-BUILD | Only rat WAT in corpus; **rebuilt young-arm-only** to stop the deployed build pooling 61% geriatric/CR tissue. Adipocyte hole → differential-only. |
| SKMGN | [GSE137869] (muscle -Y) — **shared w/ SKMVL** | `--title-include '-Y…'` → 16,30; `--label-scheme muscle` | snRNA · SD | NEEDS-BUILD | **Replaced deployed [GSE184413]** to unify muscle on one shared study (Geyu); snRNA captures myonuclei. Cost: drops the only F344 ref → head-to-head pending. |
| SKMVL | [GSE137869] (muscle -Y) — **shared w/ SKMGN** | `--title-include '-Y…'` → 16,30; `--label-scheme muscle` | snRNA · SD | NEEDS-BUILD | **Replaced the indefensible [GSE254371]** (52% rat-mouse chimera, iPSC study) under 4 top hotspots; unified with SKMGN on shared Ma -Y. |
| BLOOD | [GSE285476] (PBMC) | deployed 1-donor PBMC; immune-composition/differential | scRNA · BN+Lewis | **READY** | **Kept** — Geyu: PBMC is the field-standard whole-blood deconvolution basis (immune composition). CDSeq = cross-check; erythroid/granulocyte omission is a documented absolute-fraction caveat. |
| BAT | [GSE244451] | subscapular BAT snRNA (sample1,3) | snRNA · Dahl SS | NEEDS-BUILD | **Chosen over Ma** — subscapular-BAT snRNA recovers brown adipocytes; overrides the Ma-2020 shared source, whose scRNA has the adipocyte hole. |
| HYPOTH | [GSE248413] | `--title-include 'Y'` → sample0 (young) | snRNA · Brown Norway | NEEDS-BUILD | **Chosen** as the only rat hypothalamus in corpus; young control arm (title 'Y'). Rejected Ma whole-brain (not region-specific). |
| SMLINT | [GSE272055] | proximal jejunum (sample0,1) | scRNA · SD | NEEDS-BUILD | **Chosen** as the only clean rat small-intestine scRNA. Epithelium-only isolation (no immune/muscularis) → differential. |
| LUNG | [GSE273062](vehicle+normoxia) + [GSE252844](Control); **drop [GSE242310]** | control arms only, pooled | scRNA · SD/outbred | NEEDS-BUILD | Native pooled disease-model **control** arms; **replaced the engineered in-vitro [GSE178405]** (root cause of weak lung). **Drop ALL of [GSE242310] — BOTH its arms are P10 neonates** (StudyDesc: NOX/HOX P1–P10). |
| HIPPOC | [GSE295314] (vHPC, Fed) *[PROPOSED]* | native non-Tg | snRNA · n/s | NEEDS-BUILD | **Proposed replacement for [GSE305314]** (tauopathy-WT; a debris cluster mislabelled "monocytes" ate 64% of θ). Native non-transgenic ventral HPC — Geyu sign-off. |
| SPLEEN | [GSE186158] (GSM5639495) | rat sample of a **7-species** atlas; download+ingest | scRNA · SD | NEEDS-ONLINE-DATA | **Chosen** — only real rat spleen scRNA (online); corpus "spleen" rows were mis-mapped thymus. **Organism-gate to the rat GSM; immune-cell-isolated (NOT whole spleen — no RBC/red-pulp/stroma)** → differential/immune-composition only. |
| TESTES | [OMIX767] (C+E7W) | normal-T; **BUILT** 5,848 cells, 6 types | scRNA · SD | NEEDS-BUILD | **Chosen + built** — pool C+E7W (Geyu). Annotated with the **source paper's own DEG panel** (per-cell + marker gate): full germ lineage (SPG/SPC/RSPT/ESPT/CSPT) + Sertoli. Leydig/immune/endo too thin (dropped) → germ-stage resolution good, somatic fractions differential-only. |
| COLON | [GSE326856] (**control arm only**) | "normal rats"; verify epithelium | scRNA · SD? | NEEDS-ONLINE-DATA | Both corpus colon studies **disqualified** (DSS-colitis drug trial + ILC-sort). Online candidate is an **IBS-D model → restrict to the "normal rats" control arm**; colon-sigmoid **mucosa only** (no muscularis); verify epithelium (`Epcam+`) present. |
| ADRNL | none | reference-free CDSeq if needed | — | **BLOCKED** | **No rat adrenal scRNA/snRNA exists anywhere** (mouse/human/snATAC only); Ma-2020 has no adrenal. |
| OVARY | none ([CRA008987] spatial-only) | reference-free CDSeq interim | — | **BLOCKED** | The one adult-rat ovary scRNA (Wu 2023) is **spatial-only** in public deposits; the scRNA library was never released. Ma-2020 has no ovary. |

Roll-up: **READY 3** (KIDNEY, LIVER, BLOOD) **· NEEDS-BUILD 11 · NEEDS-ONLINE-DATA/ingest 3 · BLOCKED 2**.
Exact multi-ID sets were re-verified against the inventory. **SKMGN and SKMVL now build from the SAME
2 Ma-2020 muscle -Y samples (16,30) with the same muscle merge → one identical reference serves both muscles**
(Geyu's shared-source preference; see §3).

**Arm / age / organism restrictions — cross-checked against `deconvolution/StudyDescriptions.docx` (2026-07-15).**
The GEO descriptions confirm the selection; every multi-arm study is restricted to its control/young arm:
- **Control-arm restriction (drop the disease/treatment arm):** KIDNEY `--conditions "No treatment"` (drops
  puromycin-nephrosis + FMD arms); BLOOD `--conditions "healthy control group"` (drops the transplant *rejection*
  and *syngeneic* arms — the study is a liver-allograft model; healthy control = **1 donor**); COLON → "normal
  rats" arm (drops the IBS-D model); LUNG uses only the disease-model *control* arms (GSE273062 vehicle+normoxia,
  GSE252844 Control).
- **Age / aging-arm restriction (pick the young arm):** Ma-2020 `--title-include '-Y'` (drops old + caloric-
  restricted); HYPOTH `--title-include 'Y'` (drops 30-mo control **and** the 17α-estradiol-treated arm — note
  condition_resolved="no treatment" for BOTH young and old, so the title filter is mandatory).
- **Organism gate (multi-species studies — MUST restrict to rat):** CORTEX GSE303115 (a **6-species** cortex
  multiome) and SPLEEN GSE186158 (a **7-species** immune atlas). The default `--organism "Rattus norvegicus"`
  handles both.
- **Age skew, accepted:** several refs are young-adult vs the ~6-mo PASS1B bulk — Ma young ≈5 mo, HYPOTH young
  4 mo, HEART/VENACV 17 wk (~4 mo), LIVER 8–10 wk, SMLINT ~7–8 wk. These are the youngest healthy arms available;
  cell-type GEPs are stable across young adulthood, so age stays a tie-breaker, not a veto (no aging study offers
  a ~9-mo arm; the young arm is always the right pick).
- **Not native-composition (differential/immune only):** LIVER excludes the 2 immune-enriched scRNA samples (uses
  the 4 snRNA); SPLEEN GSE186158 is immune-cell-isolated (no red pulp/RBC/stroma); COLON GSE326856 is mucosa-only.

---

## 3. Shared-source consolidation (Geyu's one-source preference)

- **GSE137869 (Ma-2020, Sprague-Dawley) — primary for 3, always the young `-Y` arm** (arm lives only in
  geo_title): **SKMGN + SKMVL** (ONE shared skeletal-muscle reference for both muscles, per Geyu — the two
  builds are identical, so they share one reference dir) and **WATSC** (differential-only, adipocyte hole).
  Comparator, not primary, for BAT/KIDNEY/LIVER/CORTEX/HIPPOC/HYPOTH.
- **GSE280111 (Wistar cardiovascular snRNA atlas) — primary for 2:** **HEART** (LV, 19 GSMs) and **VENACV**
  (pulmonary-vein subset, 8 GSMs — the best venous proxy).
- **BLOOD keeps GSE285476 (PBMC), standalone** — per Geyu, PBMC is the field-standard whole-blood
  deconvolution basis (immune composition; the erythroid/granulocyte omission is a documented limitation on
  absolute fractions, with CDSeq as a cross-check). Ma-2020 bone marrow is NOT used.
- **Where a shared source is DISQUALIFIED for a tissue** (rule-4 defect clause): *adipocyte hole* → BAT uses
  GSE244451 snRNA (Ma scRNA recovers no adipocytes); *whole-brain non-region-specificity* → CORTEX/HIPPOC/HYPOTH
  use region-specific studies; *assay + mRNA-bias* → KIDNEY (proximal tubule) & LIVER (hepatocytes) keep the
  deployed **snRNA**. snRNA is the correct choice *because* of BayesPrism's mRNA bias, not despite it: it both
  (a) captures the fragile parenchyma that scRNA dissociation loses/degrades, and (b) *reduces* the mRNA-content
  bias — nuclear RNA is far more uniform across cell types than whole-cell cytoplasmic mRNA, so the estimated
  mRNA-fraction lands closer to the true cell-fraction. A parenchyma-poor scRNA ref does NOT counteract the bias
  (BayesPrism sets fractions from the bulk mixture under a ~uninformative prior, not from reference proportions);
  it just misassigns the bulk parenchyma mRNA to surrogate types (the omitted-component failure). So Ma's *scRNA*
  kidney/liver would move the wrong way on both axes — the one place I still recommend against Geyu's Ma
  suggestion; his call. Ma-2020 covers none of ADRNL/COLON/HEART/LUNG/OVARY/SMLINT/SPLEEN/TESTES/VENACV.
- **Cost of unifying muscle on Ma:** we drop GSE184413 (F344/BN) — the panel's ONLY strain-exact reference and
  the only *validated* gastrocnemius ref (omnideconv panel). Ma muscle -Y is snRNA (captures the myonuclei that
  dominate bulk muscle mRNA — the right assay) but is unvalidated here. Confirm the switch with a head-to-head on
  real F344 gastrocnemius bulk (GSE184413 vs Ma -Y) before trusting SKMGN absolute fractions.

---

## 4. Decisions

**RESOLVED per Geyu (2026-07-15):**
- **MUSCLE — unify both muscles on the shared GSE137869 muscle `-Y`.** Geyu grouped muscle with kidney/liver
  under GSE137869 and asked to keep one study across tissues; the earlier asymmetric proposal (SKMGN on
  GSE184413, SKMVL on Ma) did not honor that. Both SKMGN and SKMVL now use the SAME Ma-2020 muscle `-Y`
  reference (snRNA, which correctly captures myonuclei). Cost: drops GSE184413 (the only F344-strain + only
  validated gastroc ref) → run a head-to-head on real F344 gastroc bulk as a confirmation (not a gate).
- **BLOOD — keep GSE285476 (PBMC).** Geyu: PBMC is the field-standard whole-blood deconvolution basis. Reported
  as immune composition / differential; CDSeq is a cross-check, and the erythroid/granulocyte omission is a
  documented limitation on absolute fractions (not a reference switch).

**STILL OPEN:**
1. **HIPPOC reference switch.** Replace the deployed GSE305314 (tauopathy-WT; a debris cluster mislabelled
   "monocytes" ate 64% of θ) with GSE295314 ventral-hippocampus "Fed" arm (native, non-transgenic). Caveats:
   ventral-HPC only, acute 24 h food-deprivation design, strain not stated, a suspicious "Mmul10" line in the
   GEO deposit (likely a copy-paste error). **Confirm the switch, or keep GSE305314-WT with the debris cluster dropped.**
2. **Scope of BLOCKED tissues.** ADRNL and OVARY have **no rat reference anywhere** (OVARY's one adult scRNA study
   is spatial-only in public deposits). Document-and-defer, or reference-free CDSeq this round?
3. **COLON.** The only online candidate (GSE326856) looks flow-sorted with epithelium excluded — verify `Epcam+`
   in the downloaded matrices; if absent, COLON joins the no-reference class.
4. **WATSC depot.** Ma-2020 never states subcutaneous vs visceral. If visceral, WATSC is mapped to the wrong depot.
   Its absolute composition and apparent sex-composition difference are **adipocyte-hole artifacts** regardless.

---

## 5. Missing-tissue plan

- **TESTES** — 2-phase: (1) ingest staged OMIX767 arms **C + E7W** (`data/raw/ngdc/datasets/OMIX767/Gene_bc_matrices/{C,E7W}_gene_bc_matrices/`, 2,157 + 3,693 = 5,850 cells): mtx→h5ad, ambient-correct (SoupX manual — filtered matrices only, no CellBender), QC/cluster/annotate, append 2 inventory rows; (2) `build_reference.py --study OMIX767 --sample-ids OMIX767_sampleC,OMIX767_sampleE7W`. Spermatids recovered (passes parenchyma bar); heavy ambient + thin somatics ⇒ differential-only. Evaluate **GSE268104** (rat snRNA multiome, unselected fraction) as a cleaner upgrade first.
- **SPLEEN** — ingest online **GSE186158** (GSM5639495; verified rat spleen 10x, healthy, ~10k cells); confirm age/sex/strain (proposed 6-wk juvenile male SD, pooled). ACK RBC-lysis + live-sort removes red-pulp erythroid ⇒ cross-check CDSeq, frame differential. Ma-2020 bone-marrow `-Y` is the in-corpus fallback proxy.
- **VENACV** — build the GSE280111 pulmonary-vein subset (8 dedup GSMs); drop the myocardial-sleeve cardiomyocyte cluster post-build. No rat vena cava exists anywhere (only IVC atlas GSE221978 is mouse). Differential-only.
- **ADRNL** — no rat adrenal scRNA/snRNA exists (only mouse GSE108097, human GSE134355; rat snATAC atlas excludes adrenal, wrong modality). Leave non-deconvolved; CDSeq on the 50 bulk samples if a composition is required.
- **OVARY** — Wu 2023 (CRA008987/OMIX002408) has ONLY spatial runs public; the ~15k-cell scRNA library is not deposited. Document-and-defer; monitor/contact authors; CDSeq interim.

---

## 6. Execution sequencing (when decisions land)

1. **Done:** organism/title/dedup filter shipped + verified.
2. Rebuild the corpus tissues in one batch (CORTEX, HEART, VENACV, WATSC, **SKMGN+SKMVL from one shared Ma-2020
   muscle -Y build**, BAT, HYPOTH, SMLINT, LUNG-drop-neonate, HIPPOC-if-approved) with the exact commands in §2.
   BLOOD stays as-deployed (GSE285476 PBMC) — no rebuild.
3. Ingest the online/staged tissues (TESTES/OMIX767, SPLEEN/GSE186158, COLON/GSE326856-if-usable).
4. Re-run Stage 8 → 9 → 10 → 12 → 13/14 **together** — every reference change perturbs the globally-pooled
   IHW/repfdr fit across all 185 pseudobulk-DE blocks, so adjusted-p shifts in every tissue. Then re-execute the
   per-stage notebooks and hand-edit their display constants.
5. Extend `tissue_references.yaml` with per-tissue `{organism, conditions, title_include, title_exclude,
   sample_ids, dedup_gsm, label_scheme, gene_join}` so every build above is reproducible from the map, and update
   `reference_dir` to the rebuilt paths.

**Untrustworthy for ABSOLUTE fractions even after rebuild** (differential/across-condition claims only; cross-check
MuSiC/SCDC + self-consistency): BLOOD, WATSC, VENACV, HEART (θ saturated ~99.6% CM), TESTES, SPLEEN, HYPOTH, BAT,
CORTEX/HIPPOC rebuilds, KIDNEY (dominant proximal tubule).

---

## Study accessions (repository links)

Chosen references and the studies they replace. GEO = NCBI GEO; OMIX/CRA = NGDC/CNCB (China National GenBank).

[GSE240658]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE240658
[GSE220075]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE220075
[GSE303115]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE303115
[GSE280111]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE280111
[GSE137869]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE137869
[GSE285476]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE285476
[GSE244451]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE244451
[GSE248413]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE248413
[GSE272055]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE272055
[GSE273062]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE273062
[GSE252844]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE252844
[GSE242310]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE242310
[GSE295314]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE295314
[GSE186158]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186158
[GSE326856]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE326856
[OMIX767]: https://ngdc.cncb.ac.cn/omix/release/OMIX767
[CRA008987]: https://ngdc.cncb.ac.cn/gsa/browse/CRA008987
[GSE305314]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE305314
[GSE254371]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE254371
[GSE184413]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE184413
[GSE178405]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178405
[GSE155699]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE155699
