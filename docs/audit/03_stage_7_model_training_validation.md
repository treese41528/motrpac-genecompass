# Part 3. Stage 7: the model, its continued pretraining on rat (retrain v2), and the Aim-1 validation

> One part of the methods audit of 2026-11-11; start at [`README.md`](README.md). The main findings of every
> part are collected in [`FINDINGS.md`](FINDINGS.md) and every reference in [`REFERENCES.md`](REFERENCES.md).
> Citations are `path:line` at commit `8249506` unless the text says otherwise.

Audit date 2026-11-11, branch `aim2b-perturbation-engine`. All paths are relative to
`/depot/reese18/apps/motrpac-genecompass` unless they start with `/`. `<hf>` stands for the installed Hugging Face
library, `/depot/reese18/apps/motrpac-env/lib/python3.12/site-packages/transformers`.

**Where the evidence lives.** Three kinds of record are cited below, and they differ in how well they are preserved:

| Record | Examples | In git? |
|---|---|---|
| Code, configs, pre-registration and results documents | `finetune/genecompass/*.py`, `finetune/genecompass/configs/*.yaml`, `docs/RETRAIN_V2_*.md`, `docs/repro_stages1_7/` | yes |
| Model files and result tables | `data/models/**`, `data/results/retrain_v2/**`, `data/training/**`, `reports/**` | no (`.gitignore:2`, `.gitignore:9`) |
| Run logs | `logs/*.out|err`, `wandb/run-*/`, `slurm/` | no (`.gitignore:5`, `.gitignore:52`; `slurm/` is local-only) |

`config/deposit_md5.tsv` records md5 sums for the v2 initial checkpoint and the v2 final model only (`config/deposit_md5.tsv:4370-4375`). It has no entry for the phase-1 model, the training mix, the probe, or any file under `data/results/retrain_v2/` or `reports/aim1_validation_v2/` (0 matching lines). The numbers behind checks C1-C5 and the Aim-1 report therefore exist only on this machine's disk and in the prose of `docs/RETRAIN_V2_RESULTS.md`.

**Abbreviations.** MLM = masked language modelling (predict hidden tokens). CLS = the extra token placed in front of each cell's gene sequence; its final hidden state is used as the cell embedding. LR = learning rate. fp16 = 16-bit floating-point mixed precision. DDP = PyTorch distributed data parallel (one process per GPU). HF = Hugging Face `transformers`. W&B = Weights & Biases logging. kNN = k nearest neighbours. ARI = adjusted Rand index. AUC = area under the ROC curve.

**Token tiers** (from `data/training/ortholog_mappings/rat_token_mapping.tsv`, column `tier`; counts by `awk` on that file, 22,213 rat genes):

| Tier | Rat genes | Token used | Source of definition |
|---|---|---|---|
| T1 tri-species one-to-one | 13,178 | the human ortholog's existing GeneCompass token | `analysis/aim1_validation/build_inputs.py:125-129` |
| T2a human-only one-to-one | 705 | existing human token | `analysis/aim1_validation/analyze.py:127-128`; tier labels as in `docs/RETRAIN_V2_RESULTS.md:651-656` |
| T2b mouse-only one-to-one | 3,613 | existing mouse token | same |
| T3a human one-to-many | 1,351 | NEW token (id >= 50,558), cloned from the human ortholog | `finetune/genecompass/rat_model_init.py:91-107` |
| T3b mouse one-to-many | 892 | NEW token, cloned from the mouse ortholog | same |
| T4 rat-only (no ortholog call) | 2,474 | NEW token, random | `finetune/genecompass/rat_model_init.py:145-157` |

New tokens: 2,243 (T3) + 2,474 (T4) = 4,717; vocabulary 50,558 -> 55,275 (`logs/init_v2_11910364.err:4-5`, `:10`).

**The four models compared in this section** (`docs/RETRAIN_V2_PREREGISTRATION.md:19-24`):

| Name | Path | What it is |
|---|---|---|
| base | `vendor/GeneCompass/pretrained_models/GeneCompass_Base` | upstream human + mouse GeneCompass |
| April | `data/models/rat_genecompass_finetuned/models/rat_phase2_mixed_species/checkpoint-147941` | the first rat model (April 2026); superseded |
| v2 init | `data/models/rat_genecompass_init_v2` | base extended to the rat vocabulary, before any training |
| v2 | `data/models/rat_genecompass_finetuned_v2/models/rat_phase2_mixed_species/models` | phase-2 final save; the current default (`config/pipeline_config.yaml:364`) |

---

## Stage 7a - The base model (GeneCompass) and its architecture

### Research role
Question: what pretrained model is being extended to rat, and what exactly does it compute? Every downstream embedding (Stages 9, 12, 13) is a forward pass through this architecture, so the claim "pseudo-cells are embedded by a rat-adapted GeneCompass" rests on the description here. Check: the architecture is read from the vendored upstream code and the checkpoint's `config.json`; the parameter count implied by that reading is compared with the count the training job logged. Verdict: the two agree exactly (136,229,358; see Checks).

### Inputs and outputs
| Item | Path | Size / count | Source of the number |
|---|---|---|---|
| Upstream code | `vendor/GeneCompass` (git submodule, `https://github.com/xCompass-AI/GeneCompass`) | commit `66ab82f90fa94f88613df7d89a2fb38165e2a82f` | `.gitmodules:1-4`; `git submodule status` |
| Base checkpoint | `vendor/GeneCompass/pretrained_models/GeneCompass_Base/pytorch_model.bin` | 1,152,161,193 bytes, md5 `77719f3af27b3d3359032bf4ae315652`, 234 tensors | `config/genecompass_assets_md5.tsv:15`; `logs/init_v2_11910364.err:3` |
| Base config | `.../GeneCompass_Base/config.json` | 850 bytes, md5 `079a73a3...` | `config/genecompass_assets_md5.tsv:13` |
| Base vocabulary | `vendor/GeneCompass/prior_knowledge/human_mouse_tokens.pickle` | 50,558 tokens = 23,113 human + 27,443 mouse + `<pad>` (0) + `<mask>` (1) | `vendor/.../GeneCompass_Base/config.json:32`; `docs/RETRAIN_V2_PREREGISTRATION.md:180,185`; `finetune/genecompass/eval_mlm.py:38` |
| Human-mouse homolog map | `vendor/GeneCompass/prior_knowledge/homologous_hm_token.pickle` | 15,927 entries | `logs/init_v2_11910364.err:14` |
| Download source | ScienceDB link `https://www.scidb.cn/en/anonymous/SUZOdk1y` (browser download; not fetched by any script) | - | `vendor/GeneCompass/README.md:35-36`; `scripts/stage_genecompass_assets.sh:6-12` |

What the base was pretrained on, as stated upstream: "over 120 million human and mouse single-cell transcriptomes" (`vendor/GeneCompass/README.md:2`), also described as "100 million" further down (`vendor/GeneCompass/README.md:31`, `:46`); the download table describes GeneCompass_Base only as the 12-layer model (`vendor/GeneCompass/README.md:36`). The checkpoint carries an embedding-warmup counter of 960,000 forward steps (`logs/stage7_11917121.err:38`). The upstream pretraining recipe for this checkpoint (epochs, LR, batch) is NOT FOUND in the repo beyond the example command in `vendor/GeneCompass/README.md:86-110` (5 epochs, batch 10/GPU, LR 5e-5, 8 GPUs), which is labelled an example.

**Is the submodule unmodified upstream?** Yes for code. `git -C vendor/GeneCompass diff -- '*.py' '*.sh'` is empty. The nine files `git status` reports as modified are Git-LFS pointer files replaced by their real contents (132-134 bytes -> 3.9-125 MB each, all under `prior_knowledge/`; `git -C vendor/GeneCompass diff --stat`); untracked additions are `pretrained_models/`, `data/randsel_500w_human/`, `data/randsel_500w_mouse/`. `.gitmodules:4` sets `ignore = dirty` for this reason. Rat-specific behaviour is added from outside by monkey-patching (7b, 7c).

### Procedure (what one forward pass computes)
Input per cell: `input_ids` (up to 2,048 gene tokens in descending expression order, zero-padded), `values` (one float per token: log2(1 + normalised count / gene median); Stage 5, `pipeline/05_tokenization/tokenize_corpus.py:38-40,122`), and `species` (0 human, 1 mouse, 2 rat).

1. **Token embedding.** `word_embeddings` = `nn.Embedding(vocab, 768, padding_idx=pad)` (`vendor/GeneCompass/genecompass/knowledge_embeddings.py:65`).
2. **Expression value.** The scalar value is appended to the 768-d token embedding as one extra dimension, not passed through an encoder (`knowledge_embeddings.py:188-190`; `use_value_embed: false`, `GeneCompass_Base/config.json:30`).
3. **Four prior-knowledge embeddings.** Four fixed lookup tables, each `[vocab, 768]`, registered as buffers (not trainable): promoter, co-expression, gene family, PECA gene-regulatory-network ("peca_grn") (`knowledge_embeddings.py:74-112`). For each token the four rows are looked up and each passed through its own trainable linear layer 768 -> 768 (`PriorEmbedding`, `knowledge_embeddings.py:46-59`, `:197-212`), multiplied by a warm-up factor alpha (`modeling_bert.py:1332-1348`; alpha = 1.0 in the base and every checkpoint evaluated, `data/results/retrain_v2/mlm/score_c1c2.txt:10`), and concatenated.
4. **Mouse remap for priors.** For mouse cells only (`species == 1`), token ids are passed through `homologous_index` before the prior lookup (`knowledge_embeddings.py:192-194`, built at `:146-149`). Rat cells (species 2) are not remapped.
5. **Fusion.** The concatenation (768 + 1 + 4 x 768 = 3,841 dims) goes through `Linear(3841, 768)` -> LayerNorm -> QuickGELU -> `Linear(768, 768)` (`knowledge_embeddings.py:114-120`, `:214`).
6. **Positions.** Learned absolute position embeddings for positions 0-2,047 and a token-type embedding are added, then LayerNorm and dropout (`knowledge_embeddings.py:132-138`, `:217-224`).
7. **Species token.** `cls_embedding(species)` (an `nn.Embedding(2, 768)` upstream, `modeling_bert.py:264`) is prepended as position 0 AFTER step 6, so the CLS token has no position embedding and does not pass through the embedding LayerNorm (`modeling_bert.py:405-410`); the attention mask gets a leading 1 (`modeling_bert.py:361-362`). Encoder sequence length is therefore 2,049.
8. **Encoder.** HF `BertEncoder` (imported from `transformers.models.bert.modeling_bert`, `modeling_bert.py:20-26`, `:266`), 12 layers.
9. **Heads.** Position 0 is split off as the cell (CLS) output (`modeling_bert.py:781-783`). Gene-ID head: HF `BertPredictionHeadTransform` then `Linear(768, vocab)` whose weight is tied to `word_embeddings`, plus a per-token bias (`modeling_bert.py:91-108`; tie declared in `finetune/genecompass/patches/transformers_compat.py:131-135`). Value head: the same transform type then `Linear(768, 1)` (`modeling_bert.py:111-128`, `:790`).

The cell embedding used downstream is the final-layer hidden state at position 0 (`finetune/genecompass/embed_cells.py:160-171`).

### Models and software
| Item | Value | Where pinned |
|---|---|---|
| Model class | `genecompass.BertForMaskedLM` ("Copied and modified from Huggingface modeling_bert") | `vendor/GeneCompass/genecompass/modeling_bert.py:1`, `:693` |
| Upstream target HF version | `transformers==4.30.0` | `vendor/GeneCompass/requirements.txt:23`; `GeneCompass_Base/config.json:22` |
| HF version actually used here | 4.57.6 | `wandb/run-20261010_190400-vbjj3oq6/files/requirements.txt:40`; v2 `config.json:21` |
| Project pin | `transformers>=4.38,<5.0` (a range, not an exact version) | `requirements.txt:49` |

### Parameters, thresholds and flags (architecture)
Values are from the base checkpoint's `config.json` (`vendor/GeneCompass/pretrained_models/GeneCompass_Base/config.json`, line given). The training script re-declares the same values as literals (`finetune/genecompass/rat_pretrain.py:397-419`) instead of reading them from this file or from the `model:` block of the training YAML.

| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| `num_hidden_layers` | 12 | `config.json:18`; `rat_pretrain.py:399` | encoder depth | "Same as the size of BERT base" (`vendor/GeneCompass/pretraining/pretrain_genecompass_w_human_mouse_base.py:71`) |
| `hidden_size` | 768 | `config.json:11`; `rat_pretrain.py:398` | width | same |
| `num_attention_heads` | 12 | `config.json:17`; `rat_pretrain.py:408` | heads per layer | same |
| `intermediate_size` | 3072 | `config.json:13`; `rat_pretrain.py:404` | feed-forward width | same |
| `hidden_act` | gelu | `config.json:9` | activation | none stated |
| `max_position_embeddings` | 2048 | `config.json:15`; `rat_pretrain.py:406` | genes per cell | "GeneCompass paper: top-2048 genes per cell" (`pipeline/05_tokenization/tokenize_corpus.py:122`) |
| `vocab_size` | 50,558 (base); 55,275 (rat models) | `config.json:32`; v2 `config.json:30` | token count | 7b |
| `hidden_dropout_prob`, `attention_probs_dropout_prob` | 0.02, 0.02 | `config.json:10`, `:5` | dropout | none stated |
| `initializer_range` | 0.02 | `config.json:12` | init std | none stated |
| `layer_norm_eps` | 1e-12 | `config.json:14` | LayerNorm | none stated |
| `position_embedding_type` | absolute | `config.json:20` | | none stated |
| `type_vocab_size` | 2 | `config.json:23` | token-type table (always type 0) | none stated |
| `use_values`, `use_promoter`, `use_co_exp`, `use_gene_family`, `use_peca_grn`, `use_cls_token` | all true | `config.json:31,29,26,27,28,25` | which inputs are fused | none stated |
| `use_value_embed`, `concat_values_first` | false, false | `config.json:30`, `:7` | value is a raw scalar | none stated |
| species table | 2 rows upstream; 3 rows in rat models | `modeling_bert.py:264`; `rat_pretrain.py:427-430` | species token | 7b |
| masking probability | 0.15 | `vendor/GeneCompass/genecompass/pretrainer.py:620` | share of real tokens selected | none stated |
| mask recipe | of selected tokens: 80% -> `<mask>`, 10% -> random token id drawn from `[0, len(vocabulary))`, 10% unchanged; the input value is set to -1 for the first two groups | `vendor/GeneCompass/genecompass/data_collator.py:64-77` | corruption | "80% MASK, 10% random, 10% original" (`data_collator.py:44`) |
| loss weights | 0.2 (gene ID) and 0.8 (value) | `modeling_bert.py:804` | combined loss | none stated |

**Parameter count.** Parameter count of the rat-extended model (all `nn.Parameter` entries, of which two are non-trainable warm-up scalars, `modeling_bert.py:1337-1338`), as logged by the training job: 136,229,358 (`logs/stage7_11917121.err:39`; `wandb/run-20261010_190400-vbjj3oq6/files/config.yaml:572-573`). The base model's own count is not recorded anywhere; removing the 4,717 added rows (4,717 x 768 word-embedding entries, 4,717 biases, one 768-d species row) gives 132,601,217 by arithmetic. The figures "326,847,102" (base) and "348,597,957" (extended) in `finetune/genecompass/rat_model_init.py:430-432` and `logs/init_v2_11910364.err:29-30` are labelled "parameters" but count every tensor in the saved file, including the four non-trainable prior tables (4 x 55,275 x 768 = 169,804,800 values) and a second copy of the tied decoder weight. The local SLURM header's "336M-param model" (`slurm/finetune/stage7.slurm:30`) is the same kind of number and is stale.

### Statistical procedures (the two training objectives)
For a batch, let M be the set of selected (masked) positions.
- Gene-ID objective: `id_loss` = mean over M of the cross-entropy between the gene-ID head's logits over the whole vocabulary and the true token (`modeling_bert.py:795-796`; positions outside M carry label -100 and are ignored, `data_collator.py:66`).
- Value objective: `value_loss` = mean over M of (predicted value - true value)^2 (`modeling_bert.py:799-803`).
- Combined: `loss = 0.2 * id_loss + 0.8 * value_loss` (`modeling_bert.py:804`).
Nothing here is pre-registered; it is the upstream design, used unchanged.

### Checks, validation and their results
| Check | Result | Pass rule | File |
|---|---|---|---|
| Parameter count implied by the architecture read above (word 42,451,200; four prior projections 2,362,368; fusion 3,542,784; positions 1,572,864; token-type 1,536; embedding LayerNorm 1,536; species 2,304; 12 encoder layers 85,054,464; ID head 647,403; value head 592,897; warm-up scalars 2) | sums to 136,229,358 (the encoder term assumes the standard HF BERT layer layout, which was not read from a file; the exact match with the log supports it) | equals the logged count | `logs/stage7_11917121.err:39` |
| Phase-1 trainable subset (word embeddings, species table, ID-head transform and bias) | 43,100,907 by arithmetic and in the log | equal | `logs/stage7_11917121.err:39-46` |
| Submodule code unchanged | 0 diff lines in `*.py`/`*.sh` | 0 | `git -C vendor/GeneCompass diff` (run for this audit) |

### Deviations, known defects and caveats
- The HF version differs from upstream's (4.57.6 vs 4.30.0). Four monkey-patches bridge the API change (`finetune/genecompass/patches/transformers_compat.py:30-91`); one of them changes sampling (7c).
- The value objective is close to its floor from the start: base value loss is 1.7e-4 on human and 2.0e-4 on mouse (`data/results/retrain_v2/mlm/base.json:13,20`), so the combined loss is about 0.2 x `id_loss` (0.66 for an `id_loss` of 3.3). The pre-registration's explanation is that genes are ordered by value, so a masked value is bracketed by its neighbours (`docs/RETRAIN_V2_PREREGISTRATION.md:103-104`). Relative changes in value loss (used in C2) are therefore changes at the 1e-5 scale.
- `ContinuousValueEncoder` exists in the upstream file but is not used (`knowledge_embeddings.py:13-43`).

### Methods references
- GeneCompass: Yang X, Liu G, Feng G, et al. "GeneCompass: deciphering universal gene regulatory mechanisms with a knowledge-informed cross-species foundation model." Cell Research 2024;34:830-845 (`manuscript/motrpac_genecompass.tex:1487-1489`). Upstream's own README cites the bioRxiv 2023 version (`vendor/GeneCompass/README.md:149-156`).
- Length-grouped sampling and the collator design are inherited from Geneformer: Theodoris CV, Xiao L, Chopra A, et al. Nature 2023;618:616-624 (`manuscript/motrpac_genecompass.tex:1477-1478`; upstream comment pointing to Genecorpus-30M at `vendor/GeneCompass/genecompass/pretrainer.py:632`).
- BERT architecture and the 15% / 80-10-10 masking rule: Devlin et al. 2019, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" [verified against Crossref; see REFERENCES.md].

---

## Stage 7b - Model initialisation for rat

### Research role
Question: how are rat genes that have no existing GeneCompass token given a starting point, and does the starting model still behave like the base on human and mouse? Claim resting on it: the rat model starts from GeneCompass's knowledge instead of from scratch. Check: masked-prediction loss of the untrained initial model on the fixed probe (7d), and a rebuild of the checkpoint. Verdict: the initial model equals the base on human and mouse value loss and is 2.6-2.8% worse on gene-ID loss (attributed in the pre-registration to the 4,717 extra output classes, `docs/RETRAIN_V2_PREREGISTRATION.md:102-103`); it cannot be regenerated byte for byte because its random rows were drawn without a seed.

### Inputs and outputs
| Item | Path | Count / size | Source |
|---|---|---|---|
| Base checkpoint | see 7a | 234 tensors | `logs/init_v2_11910364.err:3` |
| Rat token dictionary | `data/training/ortholog_mappings/rat_human_mouse_tokens.pickle` | 55,275 entries, no gaps | `logs/init_v2_11910364.err:4`; assertion `rat_model_init.py:280-281` |
| Tier / ortholog table | `data/training/ortholog_mappings/rat_token_mapping.tsv` | 2,243 T3 tokens mapped, 0 without a usable ortholog | `logs/init_v2_11910364.err:5`; `data/models/rat_genecompass_init_v2/provenance_rat_model_init.json:112-113` |
| Stage 6 prior pickles | `data/training/prior_knowledge/{promoter,coexp,family,grn}_embeddings.pkl` | 15,610 / 16,939 / 26,019 / 21,455 keys | `logs/init_v2_11910364.err:6-9` |
| Output | `data/models/rat_genecompass_init_v2/pytorch_model.bin` + `config.json` + `provenance_rat_model_init.json` | 1,394,697,927 bytes, md5 `e2f0a614dcabac193ed66584795d14fb`; built 2026-10-08 18:06 at commit `9620e61` (SLURM job 11910364) | `config/deposit_md5.tsv:4372`; `provenance_rat_model_init.json:3-4`; `logs/init_v2_11910364.out:1` |

### Procedure
Orchestrator step 1: `pipeline/run_stage7.py:149-208` (`run_model_init`) -> `finetune/genecompass/rat_model_init.py` `extend_checkpoint` (`:228-446`), which calls `rat_load_prior_embedding.build_knowledges_dict` (`finetune/genecompass/rat_load_prior_embedding.py:217-245`). The exact command line used is stored in `data/models/rat_genecompass_init_v2/provenance_rat_model_init.json:18-45`.

1. Load the base state dict and the rat token dictionary; vocabulary size = max token id + 1 (`rat_model_init.py:268-281`).
2. Map each T3 token to its ortholog's GeneCompass token: human ortholog for T3a, mouse ortholog for T3b (`rat_model_init.py:74-112`).
3. **Prior tables** (`rat_load_prior_embedding.py:144-194`): start from zeros `[55,275, 768]`; copy the base's rows 0-50,557 unchanged; T3 rows = the ortholog's base row in all four tables; T4 rows = the Stage 6 vector for that rat gene, except in tables listed in `t4_zero_priors`, which stay zero. A x3.9 rescale is applied only to co-expression rows taken from Stage 6 (`:175-179`).
4. **Word embeddings** (`rat_model_init.py:318-342`): rows 0-50,557 copied; each T3 row = its ortholog's row + N(0, 0.001^2) per element (`:135-137`, call at `:324-327`); each T4 row = N(0, 0.02^2) per element (`:155`).
5. Output decoder weight set to a copy of the word embeddings (tied again at model construction; `rat_model_init.py:344-350`).
6. Gene-ID bias vectors extended with zeros for new tokens (`rat_model_init.py:352-358`).
7. `homologous_index` extended with the identity for new tokens (`rat_model_init.py:360-366`).
8. Species table extended from 2 to 3 rows: rat row = mouse row + N(0, 0.02^2) per element (`rat_model_init.py:368-379`).
9. Save state dict, `config.json` (base config with `vocab_size` and `pad_token_id` updated, `:420-427`) and a provenance record (`:434-444`).
10. The model class still builds a 2-row species table, so every loader replaces it with a 3-row table before loading weights (`rat_pretrain.py:427-430`; `embed_cells.py:124-126`; `eval_mlm.py:84-85`).

**What v2's initial checkpoint actually contains** (`logs/init_v2_11910364.err:10-13`, `:19-20`):

| Table | T3 rows non-zero (of 2,243) | T4 rows non-zero (of 2,474) | Reason |
|---|---|---|---|
| promoter | 2,243 | 0 | zero by policy |
| co-expression | 2,243 | 0 | zero by policy |
| gene family | 2,243 | 0 | all 2,474 T4 genes absent from the Stage 6 family pickle |
| GRN | 2,243 | 36 | all 2,474 keys present in Stage 6, 36 non-zero |
| word embedding | 2,243 cloned, 0 random fallback | 2,474 random | |

So 2,438 of the 2,474 T4 tokens enter the model with no prior knowledge of any kind, and Stage 6 contributes 36 rows in total to the v2 model.

### Models and software
`rat_model_init.py` and `rat_load_prior_embedding.py` at commit `9620e61` (`provenance_rat_model_init.json:4`); PyTorch (version at build time not recorded in the provenance file; the training jobs two days later used torch 2.11.0, `wandb/run-20261010_190400-vbjj3oq6/files/requirements.txt:156`).

### Parameters, thresholds and flags
| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| `t4_word_emb_init` / `--t4-init` | `random` | `finetune/genecompass/configs/rat_finetune_v2.yaml:73`; `provenance_rat_model_init.json` command | T4 word rows random (alternative `family_mean` unused) | "decision record, 2026-10-06; no warm start" (`docs/RETRAIN_V2_PREREGISTRATION.md:65-66`) |
| `t4_word_emb_std` / `--t4-std` | 0.02 | `rat_finetune_v2.yaml:74`; `provenance_rat_model_init.json:110` | std of T4 rows | none stated (equals `initializer_range`) |
| T3 noise std | 0.001 | hard-coded, `rat_model_init.py:326` (the YAML key `t3_noise_std`, `rat_finetune_v2.yaml:76`, is never read) | noise added to cloned T3 rows | "symmetry-breaking noise" (`rat_finetune_v2.yaml:76`) |
| T3 fallback std | 0.02 | hard-coded, `rat_model_init.py:140` | T3 token with no ortholog (0 cases) | none stated |
| `species_noise_std` | 0.02 | `rat_finetune_v2.yaml:78`; `provenance_rat_model_init.json:111` | rat species row = mouse + noise | none stated |
| `t3_prior_source` | `ortholog` | `rat_finetune_v2.yaml:80`; `provenance_rat_model_init.json:103` | T3 prior rows copied from the ortholog | Stage 6 stores T3 vectors under the human id while the old loader looked up the rat id, so all 1,351 T3a rows were zero (`rat_load_prior_embedding.py:33-36`) |
| `t4_zero_priors` | `[promoter, coexp]` | `rat_finetune_v2.yaml:81`; `provenance_rat_model_init.json:104-107` | T4 promoter and co-expression rows left zero | the Stage 6 vectors are not in GeneCompass's coordinate space; a pre-registered held-out-gene test of a rebuilt co-expression prior gave cosine 0.034 vs 0.035 shuffled (`rat_model_init.py:249-252`; `rat_load_prior_embedding.py:26-31`) |
| `coexp_rescale_factor` | 3.9 | `rat_finetune_v2.yaml:79`; `provenance_rat_model_init.json:108` | norm correction for Stage 6 co-expression rows | "new token co-exp norms are ~4.8 vs model expectation ~18.6" (`rat_load_prior_embedding.py:92-94`). No effect in v2: no co-expression row is taken from Stage 6 |
| `--seed` | not set for v2 (option added 2026-10-31; YAML default 0 for future builds) | `rat_model_init.py:511-519`; `rat_finetune_v2.yaml:82-86` | torch seed for all random rows | `rat_finetune_v2.yaml:82-85` |
| `--overwrite` | off | `rat_model_init.py:260-263` | refuses to replace an existing checkpoint | protects the starting point of a trained model (`:253-254`) |
| `ORIGINAL_VOCAB_SIZE` | 50,558 | `rat_load_prior_embedding.py:59` | boundary between old and new tokens | base vocabulary |

### Statistical procedures
None. The decision to zero T4 co-expression priors cites a pre-registered held-out-gene test (cosine 0.034 vs 0.035 shuffled, top-10 hit rate 0.1%; `rat_model_init.py:250-252`); that test belongs to Stage 6 and its code and result files were not opened for this section.

### Checks, validation and their results
| Check | Result | Pass rule | File |
|---|---|---|---|
| All T3 prior rows non-zero after copying | 2,243 / 2,243 in each of four tables | code raises otherwise (`rat_load_prior_embedding.py:190-192`) | `logs/init_v2_11910364.err:10-13` |
| No tensor left at the old vocabulary size | no "UNREIZED" warning in the log | warning only (`rat_model_init.py:381-385`) | `logs/init_v2_11910364.err` |
| Initial model on the probe, human / mouse | gene-ID loss 3.5323 / 3.5177 (base 3.4444 / 3.4215: +2.55% / +2.81%); value loss identical to base (1.728e-4 / 1.953e-4) | none (context for C2) | `data/results/retrain_v2/mlm/score_c1c2.txt:13-16` |
| Initial model on the probe, rat | gene-ID loss 7.1332, value loss 0.133, top-1 13.36% | none (reference point for C1b) | `score_c1c2.txt:17` |
| Rebuild (2026-10-31/11-01), twice | 231 of 234 tensors identical to v2's; the 3 that differ are the unseeded random draws (T3 noise, max abs difference 0.008; T4 rows; rat species row) | - | `docs/repro_stages1_7/REPRODUCTION.md:52`; `docs/repro_stages1_7/stage7_init_tensors.tsv` |
| C4 on the initial model | T3 rows have cosine 1.000 to their ortholog (median 0.9998 / 0.9997) and T4 symbol-matched genes sit at the null (median rank 10,976; p = 0.24) | expected by construction | `data/results/retrain_v2/c4/c4_summary.json:9-51` |

### Deviations, known defects and caveats
- **Not regenerable.** v2's initial checkpoint was built without a seed (`rat_model_init.py:511-514`; `rat_finetune_v2.yaml:83-85`). The orchestrator now passes `--seed 0` by default (`pipeline/run_stage7.py:196-199`), so a rebuild is reproducible but is a different checkpoint from v2's.
- **Config keys that do nothing.** `initialization.t3_warm_start`, `t3_noise_std` and `species_init` (`rat_finetune_v2.yaml:75-77`) are not read by any code; the behaviour they describe is hard-coded (`rat_model_init.py:324-327`, `:374`). `model.hidden_size` etc. (`rat_finetune_v2.yaml:13-17`) are likewise documentation only. `model.vocab_size` is used only by the orchestrator's prerequisite check (`pipeline/run_stage7.py:119-121`).
- **`--force` is parsed and ignored** (`pipeline/run_stage7.py:298-299`); the skip message that recommends it is at `:174`.
- **Paralog clones.** The 2,243 T3 tokens share 1,094 distinct ortholog tokens (`logs/v2_c4_12013887.out`, first result line), so groups of rat paralogs start from the same row and differ only by the 0.001 noise.
- **Initial loss on mouse-ortholog tokens is above the uniform-guess level.** Split by tier, the initial model's rat gene-ID loss is 4.2-4.5 nats on tokens with a human ortholog (T1, T2a, T3a) but 19.97 (T2b), 19.18 (T3b) and 19.20 (T4) on mouse-ortholog and new tokens (`data/results/retrain_v2/mlm_by_tier/init_v2.json:40-43` and neighbouring entries). A uniform guess over 55,275 tokens costs ln(55,275) = 10.92 nats, so the untrained model assigns these tokens less than uniform probability in rat cells. T2b tokens are existing mouse tokens with pretrained embeddings, so "rat species token = mouse + noise" did not make rat cells behave like mouse cells at initialisation. The cause is NOT FOUND in the repo (searched `docs/RETRAIN_V2_RESULTS.md:642-669`, which reports the numbers without explaining the initial level).
- **Manuscript disagrees with the log on T4 priors.** `manuscript/motrpac_genecompass.tex:296-297` and `:1269-1270` say T4 tokens receive gene-family and GRN rows. The init log shows 0 of 2,474 T4 family rows and 36 of 2,474 GRN rows non-zero (`logs/init_v2_11910364.err:12-13`). `finetune/genecompass/README.md:56-58` and `reports/pipeline_report.md:21` state the correct figure.
- `finetune/genecompass/README.md:10` says April's model had 53,032 tokens; April's checkpoint config has `vocab_size` 55,275 (`data/models/rat_genecompass_finetuned/models/rat_phase2_mixed_species/checkpoint-147941/config.json:30`). The docstring of `rat_pretrain.py:10` carries the same stale 53,032.

### Methods references
- DNABERT (the promoter prior that was rejected for T4): Ji Y, Zhou Z, Liu H, Davuluri RV. Bioinformatics 2021;37(15):2112-2120 (`manuscript/motrpac_genecompass.tex:1525-1527`).
- GeneCompass (prior-knowledge design): as in 7a.

---

## Stage 7c - Training (the mix, phase 1, phase 2)

### Research role
Question: can masked-prediction training on a mixture of rat, human and mouse cells teach the model rat expression while keeping its human and mouse knowledge? Claims resting on it: (i) v2 "learned rat" and (ii) v2 "did not forget human or mouse" (both scored in 7d), and every downstream result that uses v2 embeddings. Check: per-step training logs and the fixed-probe losses. Verdict so far: both phases completed one full pass over the mix with finite losses; training loss fell from 0.713 to 0.667 in phase 1 and from 0.668 to 0.658 in phase 2 (means of the first and last 200 logged values, recomputed below).

### Inputs and outputs
| Item | Path | Count / size | Source |
|---|---|---|---|
| Rat corpus (Stage 5) | `data/training/tokenized_corpus_v2/dataset` | 6,803,206 cells, 584 samples, 81 studies | `data/training/mixed_species_dataset_v2/provenance_build_mixed_species_dataset.json:71`; `docs/RETRAIN_V2_PREREGISTRATION.md:105-106`; `logs/v2_aim1_12015439.out` ("81 studies") |
| Human cells | `vendor/GeneCompass/data/randsel_500w_human` | 5,000,000 distinct cells, 165 arrow shards | `provenance_build_mixed_species_dataset.json:63`; `config/genecompass_assets_md5.tsv` (165 `data-` rows) |
| Mouse cells | `vendor/GeneCompass/data/randsel_500w_mouse` | 4,969,697 distinct cells, 164 of 165 shards (shard 00107 never arrived; `state.json` edited to drop it) | `provenance_build_mixed_species_dataset.json:67`; `scripts/stage_genecompass_assets.sh:20-24` |
| Training mix | `data/training/mixed_species_dataset_v2` (+ `sorted_length.pickle`) | 33,138,703 rows, 1,093 arrow files, 546 GB; built 2026-10-09 01:05 at commit `8ef7f86` | `provenance_build_mixed_species_dataset.json:59`, `:3-4`; `docs/repro_stages1_7/REPRODUCTION.md:55` |
| Initial checkpoint | `data/models/rat_genecompass_init_v2` | 7b | |
| Phase-1 output | `data/models/rat_genecompass_finetuned_v2/models/rat_phase1_freeze_mixed/{checkpoint-480000..517793, models/}` | final `pytorch_model.bin` 1,224,674,943 bytes; 5 checkpoints kept | directory listing; `rat_finetune_v2.yaml:108` |
| Phase-2 output (= v2) | `.../rat_phase2_mixed_species/{checkpoint-560000..591763, models/}` | `models/pytorch_model.bin` md5 `ae6100ff801674d49962c7731be97b85`, byte-identical to `checkpoint-591763` | `config/deposit_md5.tsv:4374`; `logs/v2_c1c2_12011656.out:7` |
| Per-step logs | `<checkpoint>/trainer_state.json` (`log_history`) | 517,793 and 591,763 per-step entries; 10,355 and 11,835 50-step entries | `data/results/retrain_v2/mlm/c1a.txt:1-2`; `grep -c` on the two files |

Composition of the mix (`provenance_build_mixed_species_dataset.json:59-72`):

| Species | Rows | Share of rows | Distinct source cells | Mean times each cell appears |
|---|---|---|---|---|
| human | 14,913,398 | 45.0% | 5,000,000 | 2.98 |
| mouse | 4,969,697 | 15.0% | 4,969,697 | 1.00 |
| rat | 13,255,608 | 40.0% | 6,803,206 | 1.95 |
| total | 33,138,703 | | | |

No MoTrPAC data is in the mix (`docs/RETRAIN_V2_PREREGISTRATION.md:15-17`: the rat corpus is 80 GEO single-cell studies plus OMIX767). No study is held out (`rat_finetune_v2.yaml:95`, `exclude_studies: []`).

### Procedure
Orchestrator `pipeline/run_stage7.py`: step 1 init (7b), step 2 mix, step 3 probe (7d), step 4 training (`:314-319`). Steps whose output exists are skipped (`:172-175`, `:215-217`, `:238-240`).

**Step 2, the mix** (`run_stage7.py:211-231` -> `finetune/genecompass/build_mixed_species_dataset.py:62-201`).
1. Load the three datasets; add `species` (0/1/2) where missing (`:132-142`); add `source` and `source_row` to every row and keep the rat rows' `study_id`, `sample_id`, `gsm`, `cell_id`, `cell_index` (`:107-148`).
2. `datasets.interleave_datasets([human, mouse, rat], probabilities=[0.45, 0.15, 0.40], seed=42, stopping_strategy="all_exhausted")` (`:153-158`). Each output row is drawn from one of the three sources with those probabilities; a source that runs out is reused, so its cells repeat (human: 14.9M rows from 5.0M cells); building stops when every source has been exhausted at least once. The mouse set (lowest probability) is the last to finish, which is why each mouse cell appears exactly once and the mouse set fixes the total.
3. Save, then write `sorted_length.pickle` = the true token count of every row in dataset order (`:176-181`).
4. Write row counts per species to a provenance file (`:183-197`).

**Step 4, training** (`run_stage7.py:251-263` -> `scripts/train_stage7.sh` -> `torchrun ... finetune/genecompass/rat_pretrain.py --config <yaml> --phase N --do_train --save_model --fp16 --gradient_checkpointing [--resume_from <latest checkpoint>]`, `scripts/train_stage7.sh:72-95`). v2 itself was launched by the local wrapper `slurm/finetune/stage7.slurm:109-121`, which issues the same `torchrun` line.
1. Read the YAML; overlay the `phase1:` or `phase2:` block on the shared keys (`rat_pretrain.py:115-148`, `:223-246`).
2. Seed Python and NumPy with 0 and torch with 42 (`:286-292`, defaults `:598-599`).
3. Load the token dictionary; vocabulary = max id + 1 (`:332-334`).
4. Load the state dict named by `init_checkpoint` (phase 1: the v2 initial checkpoint; phase 2: phase 1's final save, `rat_finetune_v2.yaml:10`, `:54`), take the four prior tables from it (`:376-389`), build the model with the literal architecture (`:397-422`), replace the species table by a 3-row one (`:427-430`), load weights with `strict=False` (`:435`), declare the tied weights (`:444-445`).
5. Freeze or not (`:459-465`). Phase 1 (`freeze_encoder: true`): everything frozen, then unfrozen again: all rows of the word-embedding table, all three rows of the species table, and the gene-ID head (transform + bias) (`:162-207`). Phase 2: nothing is frozen.
6. Build HF `TrainingArguments` (`:476-503`), load the mix, drop the source-reference columns (`:506-511`).
7. Apply the four compatibility patches (`:547-548`; `finetune/genecompass/patches/transformers_compat.py:30-91`).
8. Build upstream's `GenecompassPretrainer` (`:551-558`), which installs the masking collator (7a) and reads `sorted_length.pickle`; train, resuming if a checkpoint path was passed (`:561-568`); save the final model (`:571-574`).

**Sampling within a phase** (`transformers_compat.py:43-52`, `:94-118`). All four processes build the same order over the WHOLE mix: a random permutation from a generator seeded with `seed + epoch` (= 42 + 0), cut into "mega-batches", each sorted by descending length (HF `get_length_grouped_indices`). Batches of 16 (phase 1) or 14 (phase 2) are consecutive slices of that order; `accelerate` then hands each process distinct batches of that order (`transformers_compat.py:97-99`), so one optimisation step uses 64 or 56 distinct rows and one epoch covers every row once.

**What is trained in each phase** (from the job logs):

| | Phase 1 | Phase 2 |
|---|---|---|
| Trainable | `bert.embeddings.word_embeddings.weight` [55,275 x 768] (tied to the ID decoder), `bert.cls_embedding.weight` [3 x 768], `cls.predictions.bias` [55,275], `cls.predictions.transform.{dense,LayerNorm}`: 43,100,907 of 136,229,358 parameters (31.6%) (`logs/stage7_11917121.err:39-46`) | all 136,229,358 as the log words it (`logs/stage7_11961217.err:38`); two of these are the warm-up scalars, which stay non-trainable |
| Frozen | the four prior projections, the fusion network, position and token-type embeddings, embedding LayerNorm, all 12 encoder layers, the value head | nothing (the four prior tables are buffers, never trained in either phase) |
| Starting weights | `data/models/rat_genecompass_init_v2` (`logs/stage7_11917121.err:15`) | phase 1 `models/` (`logs/stage7_11961217.err:15`); optimiser, LR schedule and step counter start fresh |

Note that phase 1 updates the word-embedding rows of human and mouse tokens too, not only the new rat rows.

**Run history** (v2).
- Phase 1: SLURM job 11917121, one node, 4 x A100-SXM4-40GB, started 2026-10-10 19:01, finished 2026-10-15 18:39, exit 0, no resume (`logs/stage7_11917121.out:2-13`, `:528519-528520`). Launched at git commit `88df0bb` (`wandb/run-20261010_190400-vbjj3oq6/files/wandb-metadata.json`, `git.commit`).
- Phase 2: job 11961217. First attempt started 2026-10-16 19:12 (W&B run `run-20261016_191217-rr6j7zqf`) and logged up to step 474,012 before a node failure on 2026-10-21 (`docs/RETRAIN_V2_RESULTS.md:22-23`); the requeued job restarted 2026-10-22 09:31 from `checkpoint-470000` and finished 2026-10-23 11:16, exit 0 (`logs/stage7_11961217.out:3`, `:46`, `:124570-124571`). The SLURM log of the first attempt was overwritten by the requeue; its only record is the W&B file `wandb/run-20261016_191217-rr6j7zqf/run-rr6j7zqf.wandb`.

### Models and software
| Item | Value | Source |
|---|---|---|
| Python | 3.12.11 | `wandb/run-20261010_190400-vbjj3oq6/files/config.yaml:119` |
| torch | 2.11.0 | `.../files/requirements.txt:156` |
| transformers | 4.57.6 | `.../files/requirements.txt:40` |
| accelerate | 1.13.0 | `.../files/requirements.txt:51` |
| datasets | 4.8.4 | `.../files/requirements.txt:135` |
| numpy | 2.4.3 | `.../files/requirements.txt:117` |
| Trainer | upstream `GenecompassPretrainer` (subclass of HF `Trainer`) with four monkey-patches | `vendor/GeneCompass/genecompass/pretrainer.py:610-634`; `transformers_compat.py:30-91` |
| Optimiser | `adamw_torch_fused` (HF default for torch >= 2.8; not set by the project) | `wandb/.../config.yaml:446-447`; `<hf>/training_args.py:1292-1297` |
| Hardware | 1 node, 4 x NVIDIA A100-SXM4-40GB, 32 CPUs, 256 GB, 6-day limit | `slurm/finetune/stage7.slurm:6-12`; `logs/stage7_11917121.out:6` |

The same three library versions are recorded for both phase-2 launches (`wandb/run-20261016_191217-rr6j7zqf/files/requirements.txt:40,51,156`; `wandb/run-20261022_093423-5w3z3mzm/files/requirements.txt:40,51,156`).

### Parameters, thresholds and flags
"YAML" = `finetune/genecompass/configs/rat_finetune_v2.yaml`; "W&B" = `wandb/run-20261010_190400-vbjj3oq6/files/config.yaml` (phase 1) or `wandb/run-20261022_093423-5w3z3mzm/files/config.yaml` (phase 2), which record the arguments the Trainer actually received.

| Name | Phase 1 | Phase 2 | Where set | What it controls | Rationale given |
|---|---|---|---|---|---|
| `freeze_encoder` | true | false | YAML:30, :47; W&B phase 2 :291-292 | 7c table above | none stated (the two phases are described, not justified, in `slurm/finetune/stage7.slurm:21-26`) |
| peak learning rate | 1e-4 | 1e-5 | YAML:31, :48; W&B :368-369 / phase 2 :371-372 | AdamW LR | none stated |
| LR schedule | linear warm-up, then linear decay to 0 at the last step | same | YAML:37, :52; `rat_pretrain.py:490`; W&B :398-399 | | none stated |
| warm-up steps | 1,000 | 2,000 | YAML:36, :51; W&B :614-615 / phase 2 :617-618 | LR warm-up; the same number is also written into the model config as the embedding-warm-up threshold (`rat_pretrain.py:416`), where it has no effect because alpha is already 1.0 | none stated |
| batch size per GPU | 16 | 14 | YAML:32, :49; W&B :470-471 / phase 2 :473-474 | rows per process per step | none stated |
| GPUs / processes | 4 | 4 | YAML:115-116; `logs/stage7_11917121.err:13` | effective batch 64 / 56 | |
| gradient accumulation | 1 | 1 | not set by the project; HF default (`<hf>/training_args.py:887-888`); W&B :304-305 | | |
| epochs | 1 | 1 | YAML:33, :50; W&B :444-445 | one pass over the mix | "the sampler fix ... makes each phase cover the whole mix" (YAML:7) |
| optimisation steps | 517,793 | 591,763 | `trainer_state.json` (`max_steps`, `global_step`) | = ceil(33,138,703 / 64) and ceil(33,138,703 / 56) | |
| weight decay | 0.01 | 0.01 | YAML:67; W&B :616-617 | applied by HF to all weights except biases and LayerNorm (`<hf>/trainer.py:1266-1276`) | upstream example value (`vendor/GeneCompass/README.md:102`) |
| Adam beta1 / beta2 / epsilon | 0.9 / 0.999 / 1e-8 | same | HF defaults (`<hf>/training_args.py:917-919`); W&B :168-173 | | not set by the project |
| gradient-norm clip | 1.0 | 1.0 | hard-coded `rat_pretrain.py:498`; W&B :400-401 | | none stated |
| precision | fp16 (`--fp16`) | same | CLI flag `scripts/train_stage7.sh:93`; `rat_pretrain.py:478`; W&B :280-281. The YAML key `training.fp16` (YAML:68) is not read | mixed precision with gradient scaling (`scaler.pt` in the saved checkpoints) | none stated |
| gradient checkpointing | on | on | CLI flag `scripts/train_stage7.sh:94`; `rat_pretrain.py:481`; W&B :306-307. YAML:69 is not read | memory | "lower gpu memory cost" (`vendor/.../pretrain_genecompass_w_human_mouse_base.py:109`) |
| `group_by_length` | true | true | hard-coded `rat_pretrain.py:487-488` | length-grouped order | upstream |
| mega-batch size | 50 batches (800 rows) | 50 batches (700 rows) | HF default (`<hf>/trainer_pt_utils.py:592-594`), reached via `transformers_compat.py:115-118` | rows sorted by length within each mega-batch | none stated; upstream's own function allows up to 1,000 batches (`vendor/GeneCompass/genecompass/pretrainer.py:882-884`) |
| masking probability | 0.15 | 0.15 | `vendor/GeneCompass/genecompass/pretrainer.py:620` | 7a | upstream |
| loss weights | 0.2 / 0.8 | same | `vendor/GeneCompass/genecompass/modeling_bert.py:804` | 7a | upstream |
| dropout | 0.02 | 0.02 | `rat_pretrain.py:402-403` | | upstream |
| `ddp_find_unused_parameters` | true | false | `rat_pretrain.py:480`; W&B :226-227 / phase 2 :229-230 | needed when parameters are frozen | `rat_pretrain.py:472-473` |
| data-loader workers | 0 | 0 | `rat_pretrain.py:483`, `:608` | | upstream default |
| seeds | Python/NumPy 0, torch 42; Trainer `seed` 42 (HF default, also the sampler seed); `data_seed` null | same | `rat_pretrain.py:289-292`, `:598-599`; `<hf>/training_args.py:1063`; W&B :529-530, :208-209 | | `scripts/train_stage7.sh:24` |
| mix probabilities | human 0.45, mouse 0.15, rat 0.40 | (same dataset) | YAML:91-93; `provenance_build_mixed_species_dataset.json:53-55` | 7c mix | "restores the multi-species gradient equilibrium that GeneCompass was designed for" (`build_mixed_species_dataset.py:8-10`) |
| mix seed | 42 | | YAML:94; `provenance...json:56` | interleave draw | |
| mix stopping rule | `all_exhausted` | | `build_mixed_species_dataset.py:157`; `provenance...json:57` | total rows | |
| logging interval | 50 steps (combined `loss`, `grad_norm`, LR); `id_loss` and `value_loss` every step | same | YAML:105; `transformers_compat.py:68` | | |
| checkpoint interval / kept | every 10,000 steps / last 5 | same | YAML:107-108; `rat_pretrain.py:474`, `:496-497` | | |
| checkpoint format | `pytorch_model.bin` (`save_safetensors=False`) | same | `rat_pretrain.py:499` | | none stated |
| `emb_warmup_steps` | 10,000 | 10,000 | YAML:70; `rat_pretrain.py:417` | prior warm-up rate; inert (alpha = 1.0, `logs/stage7_11917121.err:38`) | |
| W&B project | `motrpac-genecompass-rat` | same | YAML:109 | logging | |
| `species_ratios` | 0.40 / 0.45 / 0.15 | same | YAML:41-44, :61-64 | none: "documentation only" (YAML:39-40) | |

### Statistical procedures
Training minimises the combined masked-prediction loss of 7a by AdamW; there is no validation split, no early stopping and no model selection: the model used downstream is the last step of phase 2. The logged `loss` is the 50-step mean of the combined loss on process 0's batches only, and the per-step `id_loss`/`value_loss` are process 0's batch values (`transformers_compat.py:58-71`; "Per-step rank-0 gene-ID loss", `docs/repro_stages1_7/REPRODUCTION.md:76`; the other three processes' batch losses are not in the saved log).

### Checks, validation and their results
Recomputed for this audit from the two final `trainer_state.json` files with `grep`/`awk` (no model loaded):

| Quantity | Phase 1 (`.../rat_phase1_freeze_mixed/checkpoint-517793/trainer_state.json`) | Phase 2 (`.../rat_phase2_mixed_species/checkpoint-591763/trainer_state.json`) |
|---|---|---|
| `global_step` / `max_steps` / `epoch` | 517,793 / 517,793 / 1.0 | 591,763 / 591,763 / 1.0 |
| per-step entries | 517,793 | 591,763 |
| 50-step `loss` entries | 10,355 | 11,835 |
| first / last logged `loss` | 1.0092 (step 50) / 0.6640 (step 517,750) | 0.6659 (step 50) / 0.6575 (step 591,750) |
| mean of first 200 / last 200 logged `loss` | 0.7132 / 0.6669 | 0.6677 / 0.6575 |
| minimum logged `loss` | 0.6604 | 0.6519 |
| peak LR and where | 9.999e-5 at step 1,050 | 9.999e-6 at step 2,050 |
| last logged LR | 8.5e-9 (step 517,750) | 2.4e-10 (step 591,750) |
| logged `grad_norm` = Infinity | 53 of 10,355 | 35 of 11,835 |
| `train_batch_size` | 16 | 14 |

The project notes' "phase 1 loss 0.713 -> 0.667" is therefore the mean of the first 200 and last 200 logged values, not the first and last value; it is confirmed.

Other checks:
| Check | Result | File |
|---|---|---|
| Every step logged once, finite losses (C1a) | PASS, see 7d | `data/results/retrain_v2/mlm/c1a.txt` |
| Final save equals last checkpoint | byte-identical | `logs/v2_c1c2_12011656.out:7` |
| Warm-up counter | 960,000 at phase-1 start, 1,477,793 at phase-2 start (= 960,000 + 517,793) | `logs/stage7_11917121.err:38`; `logs/stage7_11961217.err:37` |
| Restart of phase 1 from the same inputs (2,000 steps, job 12103825) | LR identical at 40 of 40 logged points; 50-step `loss` equal to 4 decimals at 39 of 40; per-step gene-ID loss bit-identical for steps 0-1, then differing by at most 9.2e-4 | `docs/repro_stages1_7/REPRODUCTION.md:63-83`; `docs/repro_stages1_7/stage7_train_loss.tsv` |
| Mix rebuild | 1,093 of 1,093 arrow files and `sorted_length.pickle` byte-identical | `docs/repro_stages1_7/REPRODUCTION.md:55` |

### The April model and the sampler defect
April was trained with the same hyperparameters (`rat_finetune_v2.yaml:1-2`; April's logged LR peaks are 9.996e-5 and 9.997e-6 with batch 16 and 14, from its `trainer_state.json` files) but differs from v2 in six ways:

| | April | v2 | Source |
|---|---|---|---|
| Rows seen per phase | 25% of the mix | 100% | below |
| Steps, phase 1 / phase 2 | 129,449 / 147,941 | 517,793 / 591,763 | `data/models/rat_genecompass_finetuned/models/*/checkpoint-*/trainer_state.json` (`global_step`); v2 as above |
| Length grouping | inert: April's `sorted_length.pickle` held 0 for every row | true lengths | `build_mixed_species_dataset.py:15-17` (statement in code; April's file was not opened for this audit) |
| Corpus | every QC file: 9,483,420 cells in 864 samples, including the 234 duplicate files and 48 non-rat samples that v2 drops | 584 kept files, 6,803,206 cells | `reports/pipeline_report.md:18-19`; `logs/v2_c3_join_12013932.out:3-5` |
| Gene medians (value scale) | "scheme A" | "scheme B" | `reports/pipeline_report.md:20`; `config/profiles/april.yaml:11-12` |
| Priors at init | Stage 6 lookup by rat id: all 1,351 T3a rows zero in every table; T4 promoter and co-expression rows from Stage 6 | 7b | `rat_load_prior_embedding.py:33-38`; `docs/RETRAIN_V2_PREREGISTRATION.md:22` |

**The defect.** Upstream's sampler for multi-GPU runs gives each process its own share of the data: it keeps `indices[rank : total : world_size]` (`vendor/GeneCompass/genecompass/pretrainer.py:743-751`, `:861`). Under HF 4.57 the Trainer also passes the data loader through `accelerate`, which splits batches across processes a second time. With 4 processes, each process therefore iterated over one quarter of its own quarter: an "epoch" was N / (batch x 4 x 4) steps and touched N / 4 rows (`transformers_compat.py:36-40`). The step counts agree with this: 129,449 = ceil(33,138,703 / 256) and 147,941 = ceil(33,138,703 / 224), a quarter of v2's counts (April's mix size is given as "33.1M rows" in `reports/pipeline_report.md:22`; its exact row count is not recorded in a small file).

**The fix** (commit `b460571`, 2026-10-08 17:35, before v2's phase 1): for more than one process the patch returns a single sampler over the whole dataset, identically seeded on every process, and leaves the splitting to `accelerate` (`transformers_compat.py:43-52`, `:94-118`). Evidence that it works: v2's step counts equal ceil(rows / (batch x 4)) and `epoch` reaches 1.0.

### Deviations, known defects and caveats
1. **Size and origin of the human and mouse sets.** Three documents call them "0.5M" cells and give the upstream download links of the "0.5M" row (`scripts/stage_genecompass_assets.sh:15-18`; `finetune/genecompass/README.md:8`; `README.md:192`; upstream table at `vendor/GeneCompass/README.md:52-53`). The mix's own provenance counts 5,000,000 distinct human and 4,969,697 distinct mouse cells, and `analysis/forgetting_analysis.py:95,100` labels the same folders "Human 5M" / "Mouse 5M" (the folder name "500w" is 5,000,000). Which upstream link the files came from cannot be determined from the repo; a rebuild that follows the documented links may fetch a different dataset.
2. **One mouse shard is missing.** 164 of 165 shards arrived; `state.json` was edited by hand to drop shard 00107 and the original kept as `state.json.bak` (`scripts/stage_genecompass_assets.sh:20-24`, `:56-83`). About 30,303 mouse cells (5,000,000 - 4,969,697) are therefore absent, and the mouse set's size determines the total number of rows.
3. **Cells repeat within a phase.** Because of `all_exhausted`, each human cell appears about 3 times and each rat cell about 2 times per phase (table above), each time with a mask drawn afresh when the batch is assembled (`vendor/GeneCompass/genecompass/data_collator.py:55-65`). The manuscript's wording "each species sampled until the largest corpus is exhausted" (`manuscript/motrpac_genecompass.tex:304-305`, `:1277-1278`) does not match the rule, which runs until every source has been exhausted.
4. **Sampler differs from upstream in mega-batch size** (50 vs up to 1,000 batches; table above). This changes how similar in length the rows of one step are; no rationale is stated.
5. **Non-finite gradient norms.** 53 (phase 1) and 35 (phase 2) of the 50-step log entries have `grad_norm` = Infinity. Only one step in 50 is logged, so the true number of such steps is not recorded. The committed documents do not discuss them (`grep` for `grad_norm`/`overflow` in `docs/` finds nothing); an uncommitted scratch trace exists at `tmp/overflow_trace/`. C1a checks losses, not gradients.
6. **No launch record inside the v2 run folders.** The code that copies the training YAML and writes `provenance_rat_pretrain_phase*.json` next to the checkpoints (`rat_pretrain.py:253-274`, `:361-365`) was added on 2026-11-01 (commit `c6be8aa`), after v2 finished; the v2 folders contain neither. What v2 ran with is known from the gitignored W&B files and SLURM logs cited above and from `training_args.bin`. The training-relevant blocks of the YAML are unchanged since commit `9620e61` (2026-10-08): `git diff 9620e61 HEAD` on the file shows only added `seed`, `mixing` and `probe` blocks. `rat_pretrain.py` changed since the launch commit `88df0bb` only by the launch-record code and two default paths.
7. **Hard-coded values that bypass the YAML**: the architecture literals (`rat_pretrain.py:397-419`), species-table width 768 (`:428`), gradient clip, `group_by_length`, checkpoint format (`:487-499`); fp16 and gradient checkpointing come from command-line flags while the YAML keys of the same name are ignored.
8. **Dead code**: `unfreeze_all` is never called (`rat_pretrain.py:210-220`); `finetune/genecompass/monitor.py` is imported by nothing (`grep` over the repo); `embed_cells.py:96-98` loads `rat_tokens.pickle` and never uses it.
9. **Mislabelled W&B fields**: `n_rat_cells` is the row count of the whole mix (33,138,703) and `n_t4_new_tokens` is all 4,717 new tokens, T3 included (`rat_pretrain.py:526-528`; W&B :420-425).
10. **Stale text.** `slurm/finetune/stage7.slurm:21-31` still describes LR 5e-5, about 2 epochs and 237,086 steps per epoch. `scripts/train_stage7.sh:8` says phase 1 took "~10 days over 2 jobs"; the log shows one job of 4 days 23.6 hours. `README.md:153` and `scripts/train_stage7.sh:26` say the phase-1 and phase-2 checkpoints are deposited; the deposit manifest lists only the initial checkpoint and the phase-2 final model (`config/deposit_md5.tsv:4370-4375`). `pipeline/README.md` gives Stage 7 one table row ("Fine-tuning", `:17`).
11. **`rat_finetune.yaml` is not April's configuration.** It is described as April's config (`pipeline/run_stage7.py:286`; `rat_finetune_v2.yaml:1-2`), but it was edited on 2026-10-05 and 10-06 to carry the new prior policy (`finetune/genecompass/configs/rat_finetune.yaml:72-73`; commits `916680e`, `45daf2d`), and its `save_steps: 10000` (`:78`) differs from the 125,000 recorded in April's phase-1 `trainer_state.json`.
12. **The April profile resolves to v2's weights.** `config/pipeline_config.yaml:364` pins `deconvolution.model_checkpoint` to the v2 final save, and `deconvolution/model_checkpoint.py:33-38` gives that key precedence over `genecompass_model_dir`. `config/profiles/april.yaml:16-27` overrides `genecompass_model_dir` (`:21`) but not `model_checkpoint`, and profiles are deep-merged over the default (`lib/gene_utils.py:46-53`, `:105-113`). Since the key was introduced (commits `c6be8aa` / `d38b83a`, 2026-11-01), a Stage 9, 12 or 13 run under `PIPELINE_PROFILE=config/profiles/april.yaml` loads v2's weights with April's medians, tokenization and target sum unless `--model-dir` is given. Confirmed for this audit by loading the merged configuration with that profile set: `resolve_model_dir` returns `data/models/rat_genecompass_finetuned_v2/models/rat_phase2_mixed_species/models` while `pseudocell_target_sum` is 6500. April results computed before that date are unaffected.
13. **No unit test covers Stage 7 code.** `tests/` has no test that imports the sampler patch, the init, the mix builder, `eval_mlm.py` or the scoring scripts (`grep` over `tests/*.py`); `tests/test_v2_outputs.py` checks Stage 9/12 outputs only (`:36-75`). `finetune/genecompass/smoke_test.py` still reads the April YAML (`:110`) and writes to the system temporary directory (`:205`).
14. **No ablation.** No run isolates the contribution of the priors, the mouse tokens or the two-phase schedule ("none is run", `docs/OPEN_FRONTIER.md:183`).

### Methods references
- GeneCompass pretraining objectives and collator: as in 7a.
- AdamW: Loshchilov and Hutter, "Decoupled Weight Decay Regularization", ICLR 2019 [verified against arXiv 1711.05101; see REFERENCES.md].
- Hugging Face `transformers` Trainer and `accelerate`: Wolf et al. 2020, "Transformers: State-of-the-Art Natural Language Processing", EMNLP demos [verified against Crossref; see REFERENCES.md]; versions as tabulated.
- Hugging Face `datasets.interleave_datasets`: library function, no paper cited in the repo.

---

## Stage 7d - Pre-registered acceptance checks C1-C5

### Research role
Question: did the retrained model (v2) learn rat expression without losing what GeneCompass knew about human and mouse, and do the analyses built on it hold up at least as well as with the April model (`docs/RETRAIN_V2_PREREGISTRATION.md:8-11`)? Claim resting on it: v2 is a valid replacement for April and is the default model for Stages 9-14. How it is checked: two pass/fail criteria on training itself (C1, C2) and three comparisons against April with no pass mark (C3, C4, C5) (`:26-27`). Verdict: C1 and C2 PASS (2026-10-23); v2 was adopted on that basis; C3 favours v2, C4 favours April, C5 is mixed.

**The "no held-out set" decision and what it means.** "There is no held-out study set (decided 2026-10-08): the rat corpus is small, and every study usable as a deconvolution reference must stay in training" (`docs/RETRAIN_V2_PREREGISTRATION.md:13-14`). Consequences for reading every number in 7d and 7e:
- The probe cells are training cells for every model ("in-sample cells for every model ... That is deliberate", `:37-38`). C1b measures how well v2 fits rat cells it was trained on; it is not evidence of generalisation to unseen rat cells or studies. C2 measures retention on cells that base and v2 both trained on. The results document says the same of the per-tier split: "this measures what each model learned, not how it generalises" (`docs/RETRAIN_V2_RESULTS.md:645-647`).
- C3 and the Aim-1 check V1 (7e) use cells that are in the training corpus. What training never saw is the labels (training is self-supervised, `:14-15`). Cross-validation folds in C3 are held out from the nearest-neighbour classifier, not from the encoder.
- The only inputs the model has never seen in any form are the MoTrPAC-derived pseudo-cells of Stage 9 onward (`:16-17`), which is why the pre-registration calls C5 "the real judge" (`:74`).

### Inputs and outputs
| Item | Path | Count / size | Source |
|---|---|---|---|
| Fixed probe | `data/training/eval_probe_v2/` | 6,000 cells (2,000 per species); arrow file 98,752,904 bytes | `data/training/eval_probe_v2/provenance_eval_mlm.json:38-39`; directory listing |
| Probe md5 (registered) | `data/training/eval_probe_v2.md5`; `docs/RETRAIN_V2_PREREGISTRATION.md:107-110` | arrow `90aa2d9030f4b769a3388a0b1ea808f9`, `dataset_info.json` `c6b1b65c0b5e667f07333ff624d2e3ff`, `state.json` `76fbde1fbb8cf2d289ec81efeaedf628` | re-checked at scoring time: `logs/v2_c1c2_12011656.out:3-5` |
| C1/C2 results | `data/results/retrain_v2/mlm/{base,init_v2,april,v2_phase1,v2}.json`, `c1a.txt`, `score_c1c2.txt` | job 12011656, 2026-10-23 | `logs/v2_c1c2_12011656.out:1` |
| C3 results | `data/results/retrain_v2/c3/{cells.tsv,c3_per_reference.tsv,c3_summary.json,emb/}` | 620,058 cells, 12 references | `logs/v2_c3_join_12013932.out` (last line) |
| C4 results | `data/results/retrain_v2/c4/{c4_summary.json,c4a_genes.tsv,c4b_genes.tsv}` | 255 T4 genes, 2,243 T3 tokens | `data/results/retrain_v2/c4/c4_summary.json:2-3` |
| C5 results | `data/results/retrain_v2/c5/` (`gate_summary.json`, `stage9_embed_qc.txt`, `stage12_cosine_{april,v2}.tsv`, `module_a/`) | 14 tissues, 172 blocks (July references) | `data/results/retrain_v2/c5/gate_summary.json:4` |

### Procedure
**Probe** (`finetune/genecompass/eval_mlm.py:41-65`, orchestrator step 3 `pipeline/run_stage7.py:234-248`; job 11916410, 2026-10-09, before phase 1 started on 2026-10-10). For each species, take the mix rows of that species, keep one row per distinct `source_row` (`:50`), and draw 2,000 without replacement with `numpy.random.default_rng(20261008)` (`:46`, `:52`). Saved once; the script refuses to overwrite (`:60-61`).

**Scoring one model** (`eval_mlm.py:116-167`). Load the model exactly as saved, with the warm-up factor fixed (`:68-90`). For each species and each batch of 8 cells: draw a mask from a `torch.Generator` seeded with `seed * 1,000,003 + species * 10,007 + batch_index` (`:134`), so every model gets the same masked positions; select 15% of non-pad positions; of those 80% become `<mask>`, 10% a random token id, 10% stay; the input value is set to -1 at the first two groups (`:102-113`). Run the model and accumulate, weighted by the number of masked tokens: gene-ID cross-entropy, value mean squared error, and top-1 accuracy (`:140-148`, `:157-159`). `--kind base` scores human and mouse only (`:123`).

**C1/C2 scoring** (`finetune/genecompass/score_prereg_c1c2.py`; command in `slurm/analysis/retrain_v2_c1c2.slurm:34-46`).
**C3** (`finetune/genecompass/eval_cell_embeddings.py`): `join` (`:75-128`) -> `embed` (16 array tasks, `slurm/analysis/retrain_v2_c3_embed.slurm`) -> `score` (`:208-239`).
**C4** (`finetune/genecompass/eval_gene_embeddings.py:86-154`).
**C5**: reruns of Stage 9, the Aim-2 gate, Stage 12 and Stage 13 Module A with v2; their methods belong to the sections on those stages and are only summarised here.

### Models and software
Models scored: base, v2 init, April (`checkpoint-147941`), v2 after phase 1, v2 (`slurm/analysis/retrain_v2_c1c2.slurm:34-38`). Evaluation is in 32-bit precision on one GPU (no autocast in `eval_mlm.py`; `embed_cells.py` likewise). scikit-learn (`StratifiedKFold`, `f1_score`) and SciPy (`wilcoxon`) are used in C3 (`eval_cell_embeddings.py:148`, `:186`, `:210`); installed versions on this machine are scikit-learn 1.8.0 and SciPy 1.17.1 (`importlib.metadata`, read for this audit; not recorded with the results).

### Parameters, thresholds and flags
| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| probe seed | 20261008 | `eval_mlm.py:177`; `rat_finetune_v2.yaml:101`; `provenance_eval_mlm.json:38` | which cells | none stated (the value is the registration date) |
| cells per species | 2,000 | `eval_mlm.py:176`; `rat_finetune_v2.yaml:102` | probe size | none stated |
| mask seed | 20261008 | `eval_mlm.py:185` | masked positions | same masks for every model (`docs/RETRAIN_V2_PREREGISTRATION.md:33-34`) |
| mask rate / recipe | 0.15; 0.8 / 0.1 / 0.1 | `eval_mlm.py:104-106` | corruption | "GeneCompass's own collator" (`...PREREGISTRATION.md:33`) |
| random-token range | 50,558 for human and mouse cells, 55,275 for rat cells | `eval_mlm.py:37`, `:130` | replacement ids | "fixed per species ... so every model sees identical inputs" (`eval_mlm.py:17-18`) |
| eval batch size | 8 | `eval_mlm.py:183` | also defines the mask seed sequence | none stated |
| C2 tolerance | v2 <= 1.10 x base | `score_prereg_c1c2.py:95`; `...PREREGISTRATION.md:49-50` | forgetting limit | none stated |
| C1b rule | v2 < v2 init (strict) | `score_prereg_c1c2.py:94` | learning | |
| C3 neighbours / folds / seed | k = 10, 5 folds, 20261008 | `eval_cell_embeddings.py:45` | kNN cross-validation | `...PREREGISTRATION.md:58` |
| C3 minimum cells per sample | 5 | `eval_cell_embeddings.py:45`, `:196-199` | sample-mixing metric | `...PREREGISTRATION.md:201-202` |
| C3 references | 13 listed, 12 scored | `eval_cell_embeddings.py:39-41` | which labelled cells | `...PREREGISTRATION.md:188-196` |
| C3 label source | `data/deconvolution/references_v3/<ref>/cells_meta.tsv`, column `cell_type` | `eval_cell_embeddings.py:102` | July references | `...PREREGISTRATION.md:54-56` |
| C4 null permutations / seed | 1,000 / 20261008 | `eval_gene_embeddings.py:90-91`, `:103-104` | re-pairing null | `...PREREGISTRATION.md:68` |
| C5 gate setting | 10 stratified 5-fold splits x 10,000 permutations (changed from the registered 1 split x 1,000 on 2026-10-24) | `...PREREGISTRATION.md:218-234` | hotspot calls | "with the registered single split and 1,000 permutations the seed decides April's hotspot set" (`:220-221`) |
| C5 Stage 9 target sum for v2 | 6,400 (April 6,500) | `...PREREGISTRATION.md:155-164`; `config/pipeline_config.yaml:413` | pseudo-cell value scale | April's rule re-applied to the v2 corpus |

### Statistical procedures
All five criteria are pre-registered in `docs/RETRAIN_V2_PREREGISTRATION.md` (first commit `8ef7f86`, 2026-10-08 21:51; phase 1 started 2026-10-10 19:01). Sections 1-5 of that file have not been edited since the first commit (`git diff 8ef7f86 HEAD` shows only added lines, all in the change log); implementation details of C3-C5 were added to the change log on 2026-10-22 and 2026-10-24.

- **C1a** (pass/fail). Statistic: for each phase's final `trainer_state.json`, the per-step `id_loss` and `value_loss` entries. Rule: the number of entries equals `global_step`, the step numbers are exactly 0..global_step-1, and no value is non-finite (`score_prereg_c1c2.py:30-39`). The steps of the interrupted phase-2 attempt (>= 470,000) are read from its W&B file and must be finite too (`:42-68`).
- **C1b** (pass/fail). Statistic: token-weighted gene-ID loss and value loss on the 2,000 rat probe cells. Rule: v2 strictly lower than v2 init on both (`:94`).
- **C2** (pass/fail). Statistic: the same two losses on the human and on the mouse probe cells. Rule: v2 <= 1.10 x base for all four (`:95`). No significance test; the unit is the pooled set of masked tokens (443,348 human, 452,385 mouse, 317,619 rat).
- The scorer also requires that every model scored the same number of cells and masked tokens (`:78-85`, `:101`).
- **C3** (comparison). Per reference, on L2-normalised CLS embeddings: (i) cell-type kNN: 5-fold stratified cross-validation, 10 nearest training cells by cosine, majority vote with ties to the lowest label code, macro-F1 and accuracy over all held-out predictions (`eval_cell_embeddings.py:144-162`, `:185-192`); (ii) silhouette by cell type with cosine distance over all cells, computed in closed form (`:165-182`); (iii) sample mixing: the same kNN predicting `sample` within each cell type that has at least two samples of at least 5 cells, pooled accuracy, lower = less batch-driven (`:193-205`). Summary: median over references and a two-sided paired Wilcoxon signed-rank test of v2 vs April across references (`:230-236`); the unit of replication is the reference (n = 12, or 10 for sample mixing). No multiplicity correction across the four metrics.
- **C4a** (comparison with a null). For each of the 255 T4 tokens whose upper-cased rat symbol equals the symbol of exactly one human token, the rank (1 = nearest) of that human token among all 23,113 human tokens by cosine of the input word embeddings (`eval_gene_embeddings.py:42-60`, `:73-79`, `:113-117`). Statistics: median rank, share with rank <= 10, share with rank <= 100 (`:82-83`). Null: the same statistics after permuting which human counterpart is paired with which rat token, 1,000 permutations generated once and reused for every model (`:103-104`, `:118`). One-sided p = (1 + number of null values at least as good) / 1,001 (`:122-124`); smallest possible p = 0.000999.
- **C4b** (descriptive, "drift, not success", `...PREREGISTRATION.md:70-72`). For each T3 token, cosine to the ortholog token it was cloned from and that ortholog's rank among tokens of its own species (human tokens for T3a; the 27,443 mouse tokens for T3b) (`eval_gene_embeddings.py:127-136`).
- **C5** (comparison). Stage 9: cosine silhouette by cell type per tissue. Gate: per tissue x cell-type block, cross-validated PLS-1 probe AUC for trained vs control with a permutation p-value; hotspots = Benjamini-Hochberg q < 0.05 over all blocks. Stage 12: cosine between each pseudo-cell's rat-space and human-space embedding; transfer status PRESERVED (human-space AUC >= 0.65 and p < 0.05), WEAKENED (AUC >= 0.65, p >= 0.05), LOST (AUC < 0.65) (`docs/RETRAIN_V2_RESULTS.md:302-306`). Module A: four frozen perturbation criteria. Definitions: `docs/RETRAIN_V2_PREREGISTRATION.md:74-85`, `:205-251`.
- **Decision rule** (`...PREREGISTRATION.md:87-90`): C1 or C2 fails -> v2 not adopted. C1 and C2 pass -> v2 becomes the working model; C3-C5 are reported "including any where v2 is worse than April".

### Checks, validation and their results
**C1 and C2** (`data/results/retrain_v2/mlm/score_c1c2.txt:1-35`; identical figures in `docs/RETRAIN_V2_RESULTS.md:31-54`):

| Model | Species | Gene-ID loss | Value loss | Top-1 | Gene-ID vs base | Value vs base |
|---|---|---|---|---|---|---|
| base | human | 3.4444 | 1.728e-4 | 21.25% | | |
| base | mouse | 3.4215 | 1.953e-4 | 21.48% | | |
| v2 init | human | 3.5323 | 1.728e-4 | 20.58% | +2.55% | 0.00% |
| v2 init | mouse | 3.5177 | 1.953e-4 | 20.71% | +2.81% | 0.00% |
| v2 init | rat | 7.1332 | 1.330e-1 | 13.36% | | |
| April | human | 3.4495 | 1.724e-4 | 21.23% | +0.15% | -0.24% |
| April | mouse | 3.4545 | 2.018e-4 | 21.21% | +0.96% | +3.33% |
| April | rat | 5.3102 | 3.916e-4 | 18.80% | | |
| v2 after phase 1 | human | 3.4634 | 1.740e-4 | 21.10% | +0.55% | +0.67% |
| v2 after phase 1 | mouse | 3.4854 | 2.094e-4 | 20.97% | +1.87% | +7.24% |
| v2 after phase 1 | rat | 3.1565 | 5.421e-4 | 24.27% | | |
| **v2** | human | 3.4462 | 1.680e-4 | 21.22% | +0.05% | -2.81% |
| **v2** | mouse | 3.4667 | 2.002e-4 | 21.22% | +1.32% | +2.53% |
| **v2** | rat | 3.0590 | 2.477e-4 | 25.51% | | |

| Criterion | Obtained | Pass rule | Verdict | File |
|---|---|---|---|---|
| C1a | 517,793 of 517,793 and 591,763 of 591,763 steps logged once, 0 non-finite; 4,013 superseded steps of the first phase-2 attempt finite | all finite, each step once | PASS | `data/results/retrain_v2/mlm/score_c1c2.txt:2-4` |
| C1b | rat gene-ID 3.059 vs 7.133; value 2.477e-4 vs 0.133 | v2 < init | PASS | `score_c1c2.txt:28-29` |
| C2 | human +0.05% and -2.81%; mouse +1.32% and +2.53% | each <= +10% | PASS | `score_c1c2.txt:30-33` |
| Same cells and masks for all models | 2,000 cells and 443,348 / 452,385 / 317,619 masked tokens for every model | identical | yes | `score_c1c2.txt:7-10` |

**C3** (`data/results/retrain_v2/c3/c3_summary.json`; per-reference rows in `c3_per_reference.tsv:2-25`). 620,058 cells of 12 references; TESTES (5,836 cells) is absent from April's corpus and drops out (`logs/v2_c3_join_12013932.out`).

| Metric | n | April median | v2 median | Median v2 - April | v2 higher | Wilcoxon p (two-sided) |
|---|---|---|---|---|---|---|
| cell-type kNN macro-F1 | 12 | 0.7713 | 0.8039 | +0.0212 | 12 / 12 | 0.000488 |
| cell-type kNN accuracy | 12 | 0.9362 | 0.9427 | +0.0089 | 12 / 12 | 0.000488 |
| cell-type silhouette (cosine) | 12 | 0.0449 | 0.0616 | +0.0397 | 12 / 12 | 0.000488 |
| sample kNN accuracy within cell type (lower = less batch-driven) | 10 | 0.8731 | 0.8893 | +0.0115 | 10 / 10 | 0.00195 |

0.000488 and 0.00195 are the smallest two-sided p-values attainable with 12 and 10 pairs. Silhouettes are negative for both models in five references (HIPPOC, KIDNEY, LIVER, LUNG, SMLINT; `c3_per_reference.tsv:10-11,14-19,22-23`); HIPPOC macro-F1 is 0.255 / 0.273 (`:10-11`).

**C4** (`data/results/retrain_v2/c4/c4_summary.json`):

| Model | C4a median rank (null mean; 2.5-97.5%) | top-10 | top-100 | p (median rank) | C4b T3a: median cosine / ortholog still nearest / in top 10 | C4b T3b: same |
|---|---|---|---|---|---|---|
| v2 init | 10,976 (11,504; 10,015-13,019) | 0.0% | 0.0% | 0.238 | 1.000 / 100% / 100% | 1.000 / 100% / 100% |
| April | 3,298 (12,466; 11,374-13,618) | 5.5% | 12.2% | 0.000999 | 0.810 / 100% / 100% | 0.801 / 96.9% / 99.8% |
| v2 | 6,593 (13,030; 12,054-13,974) | 2.7% | 4.7% | 0.000999 | 0.705 / 75.1% / 82.6% | 0.673 / 55.3% / 71.6% |

(`c4_summary.json:9-51`, `:55-97`, `:101-143`.) Both trained models place symbol-matched T4 genes nearer their human namesake than chance; April does so more strongly than v2. v2's T3 rows have moved further from the ortholog rows they were copied from.

**C5** (July references, 14 tissues, 172 blocks):

| Stage and metric | April | v2 | File |
|---|---|---|---|
| Stage 9: median over tissues of the cell-type silhouette | 0.712 | 0.731 (higher in 10 of 14 tissues) | `data/results/retrain_v2/c5/stage9_embed_qc.txt:4-17`, `:21-34` |
| Gate, C5 method (10 x 10,000): exercise hotspots | 16 | 13 (11 shared) | `data/results/retrain_v2/c5/gate_summary.json:90`, `:114`, `:157` |
| Gate, C5 method: paired trained AUC, median v2 - April; blocks where v2 is higher | | -0.0015; 84 of 172 | `gate_summary.json:137-141` |
| Gate, registered method (1 x 1,000): hotspots | 7 | 11 | `gate_summary.json:5`, `:29` |
| Stage 12: median over tissues of the mean rat-human cosine | 0.973 | 0.969 | `data/results/retrain_v2/c5/stage12_cosine_{april,v2}.tsv` (median of column `mean`, recomputed) |
| Stage 12 transfer of own hotspots, C5 method: preserved / weakened / lost | 14 / 0 / 2 | 13 / 0 / 0 | `gate_summary.json:95-99`, `:119-123` |
| Stage 12: Spearman of trained AUC, rat vs human space, 172 blocks | 0.762 | 0.616 | `gate_summary.json:107`, `:131` |
| Module A, four criteria | 4 of 4 pass | 3 of 4 pass; #2 fails (0 of 25 top targets in the curated set, p = 1.0) | `data/results/retrain_v2/c5/module_a/{april,v2}/report.md:7-10` |

**Decision.** "C1 and C2 pass, so v2 is the working model for the downstream reruns" (`docs/RETRAIN_V2_RESULTS.md:59`, 2026-10-23). v2 became the default configuration on 2026-10-25 (commits `703eec3`, `311f036`; `config/pipeline_config.yaml:364`, `:410`, `:413`); April is reachable through `config/profiles/april.yaml` (but see 7c caveat 12).

### Deviations, known defects and caveats
1. **Adoption rested on C1 and C2 only**, both of which are in-sample training-loss checks. The comparisons that came afterwards are not uniformly in v2's favour: C4a (April nearer the human namesakes), fewer hotspots under the C5 method (13 vs 16), Module A criterion #2 fails for v2, and the rat-to-human ordering of blocks is weaker (0.616 vs 0.762). The results document also reports, outside C5, lower GRN self-consistency for v2 (median rho 0.292 vs 0.451, `docs/RETRAIN_V2_RESULTS.md:499`). The pre-registered rule allows this (`...PREREGISTRATION.md:89-90`); a reviewer should not read "adopted" as "better on every measure".
2. **The 10% tolerance cannot be shown to pre-date the first look at the numbers.** The first committed version of the pre-registration already contains, in its change log, an evaluator smoke test on a non-registered probe that reports base, v2-init and April losses and notes "April would pass C2 (+0.2% to +3.3%)" (`docs/RETRAIN_V2_PREREGISTRATION.md:98-104`; `git show 8ef7f86`). The document states "No criterion was changed after seeing these" (`:104`); git cannot confirm or refute it. No v2 result existed at that time.
3. **The masking is a re-implementation, not upstream's collator object.** It follows the same 15% / 80-10-10 rule and sets values to -1 in the same cases (`eval_mlm.py:102-113` vs `vendor/GeneCompass/genecompass/data_collator.py:64-77`), but draws random replacement ids from the 50,558-token range for human and mouse cells, whereas the training collator draws from the whole 55,275-token vocabulary for every cell (`data_collator.py:75`). The manuscript's "scored with GeneCompass's own masking collator" (`manuscript/motrpac_genecompass.tex:1283`) is therefore loose. PAD and MASK ids and the rat vocabulary size are literals (`eval_mlm.py:37-38`, `:130`).
4. **C1a sees process 0 only.** The per-step losses are those of one of the four processes (7c, Statistical procedures). The agreement of the first phase-2 attempt's W&B steps below 470,000 with the final log (470,000 of 470,000) is what one would expect if the resumed run inherits the log history stored in `checkpoint-470000`; this was not verified in HF code for this audit.
5. **April is scored on values it was not trained on.** The probe holds scheme-B values; April was trained on scheme-A values, so its rat numbers are "context, not a target" (`...PREREGISTRATION.md:45-46`). The same applies to April's per-tier losses in 7e.
6. **C3 uses the July reference labels** (`references_v3`, `eval_cell_embeddings.py:102`), not the jointly re-annotated references that became the default on 2026-10-31. C3 has not been recomputed on the current labels.
7. **Code changed after the "implementation fixed" log entries, before any metric existed.** The first C4 job (12013862) and the first C3 join job (12013889) crashed (`logs/v2_c4_12013862.err`, `logs/v2_c3_join_12013889.err`); four small fixes followed within about 20 minutes (commits `2dc100c`, `029d328`, `ea0034e`, `b5d0874`). The crashed jobs printed no metric.
8. **By the registered reading, v2 is the more batch-driven model in C3**: sample identity is easier to predict within a cell type in all 10 references where it is defined. The results document notes the metric cannot separate batch from real between-sample differences (`docs/RETRAIN_V2_RESULTS.md:113-117`).
9. **The gate method was changed after registration.** The change is logged before any v2 gate number existed (`...PREREGISTRATION.md:218-219`), but it was motivated by April's seed sensitivity, and the two methods order the models oppositely (registered: v2 11 vs April 7; C5 method: 13 vs 16). Both are reported.
10. **Module A criterion #2 is not neutral between models**: the curated gene set was written after April's smoke run and includes genes that run surfaced (`docs/RETRAIN_V2_RESULTS.md:359-363`).
11. **C5 is frozen on superseded inputs.** It was scored on the July references; the current default is v2 on the re-annotated references and 16 tissues, with different hotspot counts (`docs/RETRAIN_V2_RESULTS.md:3-13`). The pre-registration says the C5 comparison "stays as scored" (`...PREREGISTRATION.md:281-282`). No April embedding folder exists for the current references (the only April folders under `data/deconvolution/` are `genecompass_input/` and `genecompass_input_human/`), so April and v2 have not been compared on them.
12. **Evidence files are not version-controlled** (see the table at the top of this section).

### Methods references
- Wilcoxon signed-rank test: Wilcoxon 1945, Biometrics Bulletin 1:80-83 [verified against Crossref; see REFERENCES.md]; implementation `scipy.stats.wilcoxon` (`eval_cell_embeddings.py:210`, `:233`).
- Silhouette: Rousseeuw 1987, J Comput Appl Math 20:53-65 [verified against Crossref; see REFERENCES.md]; closed-form cosine version in `eval_cell_embeddings.py:165-182`.
- Benjamini-Hochberg false discovery rate (C5 hotspots): Benjamini Y, Hochberg Y. J R Stat Soc Series B 1995;57:289-300 (`manuscript/motrpac_genecompass.tex:1545-1547`).
- Augur (context in C5): Skinnider MA, Squair JW, Kathe C, et al. Nat Biotechnol 2021;39:30-34 (`manuscript/motrpac_genecompass.tex:1499-1500`).

---

## Stage 7e - Aim-1 validation (V1-V3, per-tier loss, forgetting trajectory, annotation tools)

### Research role
Question: beyond the training loss, are the model's representations organised the way biology says they should be: do rat cells group by cell type (V1), do rat genes sit near their human or mouse orthologs (V2), and did the brand-new rat-only tokens learn anything recognisable (V3)? The report generator describes them as the three checks promised in section 3.6 of `reports/pipeline_report.md` (`analysis/aim1_validation/analyze.py:311-312`); project notes call them the validation trio (`docs/OPEN_FRONTIER.md:99`). They are not pre-registered and are not adoption criteria; for v2 they were rerun as part of the downstream reruns (`docs/RETRAIN_V2_PREREGISTRATION.md:265-267`). Verdicts for v2: V1 PARTIAL, V2 high AUCs (called PASS in documents; the code assigns no verdict), V3 WEAK.

### Inputs and outputs
| Item | Path | Count | Source |
|---|---|---|---|
| Rat corpus | `data/training/tokenized_corpus_v2/dataset` | 6,803,206 cells, 81 studies, all with annotation folders | `logs/v2_aim1_12015439.out` ("corpus n=6803206", "81 studies, 81 annotated") |
| Cell-type labels for V1 | `data/training/cell_annotations/<sample>/<sample>_celltypes.tsv` | automated marker-based labels (below), written 2026-04-12 | `analysis/aim1_validation/build_inputs.py:39,49`; `analysis/annotate_celltypes.py:75` |
| Labelled subset | `data/validation/aim1_v2/{subset_rat/,labels.tsv,subset_summary.json}` | 40,332 cells, 44 cell types, 17 studies; 104,128 barcodes dropped for conflicting labels | `data/validation/aim1_v2/subset_summary.json`; `logs/v2_aim1_12015439.out` |
| Homolog pairs | `data/validation/aim1_v2/pairs.tsv` (+ `rat_targets.npy`, `human_targets.npy`, `mouse_targets.npy`) | 3,243 pairs: 1,000 T1, 1,351 T3a, 892 T3b; byte-identical to April's | `logs/v2_aim1_12015439.out` |
| Human / mouse context cells | first 50,000 rows of `vendor/GeneCompass/data/randsel_500w_{human,mouse}` | 50,000 each | `slurm/analysis/v2_aim1.slurm:37-41` |
| Outputs | `reports/aim1_validation_v2/{AIM1_VALIDATION_REPORT.md,metrics.json,figures/}`; embeddings in `data/validation/aim1_v2/out_{rat,human,mouse}/` | jobs 12015439 (2026-10-24) and 12016595 (report regenerated) | `docs/RETRAIN_V2_RESULTS.md:603`, `:635-640` |
| April's run | `reports/aim1_validation/` (2026-06-25/26, `checkpoint-147941`) | | `reports/aim1_validation/AIM1_VALIDATION_REPORT.md:2` |
| Per-tier loss | `data/results/retrain_v2/mlm_by_tier/{init_v2,april,v2_phase1,v2}.json` | job 12103793, 2026-10-31 | `slurm/analysis/mlm_by_tier.slurm` |
| Forgetting trajectory (April only) | `reports/forgetting/{forgetting_report.md,forgetting_summary.tsv,forgetting_results.json,encoder_drift.json}` | 11 checkpoints, generated 2026-04-12 | `reports/forgetting/forgetting_report.md:3-7` |
| Annotation tables | `reports/annotations/` (`annotation_inventory.tsv/json`, `tissue_coverage_report.tsv`, `tissue_motrpac_map.json`) | sample-level metadata | `analysis/annotation_recovery.py:12-15` |

### Procedure
Driver for v2: `slurm/analysis/v2_aim1.slurm:28-43`.

1. **Build inputs** (`analysis/aim1_validation/build_inputs.py`). V1 subset (`:64-107`): shuffle the annotated studies (`random.shuffle`, seed 42, `:80`, `:164`); take studies in that order until at least 40,000 labelled cells are collected (`:82-83`); cap each study at 2,500 randomly chosen rows (`:87-88`); give each row the label of its barcode from that study's annotation files, where a barcode that carries different labels in different samples of the study is dropped (`:47-61`); drop labels "nan"/"unknown"/empty (`:91`) and cell types with fewer than 50 cells in the subset (`:96`). V2 targets (`:110-149`): 1,000 T1 genes sampled with `random_state=42` (`:122-123`), all T3a and all T3b genes, each with its rat token and its ortholog's token.
2. **Extract embeddings** (`analysis/aim1_validation/extract_embeddings.py`). One forward pass per cell (batch 32, `:75`), species 2 for the rat subset, 0 for human cells, 1 for mouse cells. CLS embedding = final hidden state at position 0 (rat run only, `:113-114`). Contextual gene embedding of a token = mean over all cells that contain it of the final hidden state at that token's position (`:115-123`, `:132-136`).
3. **Analyse** (`analysis/aim1_validation/analyze.py`).
   - **V1** (`:57-114`). k = 15 nearest neighbours of every cell on the raw (not normalised) CLS vectors; no `metric` argument is passed to `NearestNeighbors`, so scikit-learn's default applies (Euclidean [confirmed in the project environment: scikit-learn 1.8.0, `metric = "minkowski"`, `p = 2`]) (`:68-69`). C3, by contrast, uses cosine. Cell-type purity = share of neighbours with the same label; study purity likewise; chance = sum of squared label frequencies (`:70-76`). Cross-study mixing = among neighbours with the same cell type, the share from a different study, averaged over cells whose cell type occurs in at least two studies (`:79-86`). ARI between k-means (k = number of cell types, `n_init=4`, `random_state=0`) and the labels (`:88-90`). Verdict: PASS if purity > 3 x chance and cross-study mixing > 0.30; PARTIAL if purity > 1.5 x chance; else WEAK (`:107-108`).
   - **V2** (`:118-207`). For each tier, cosine between the rat gene's embedding and its ortholog's. Null: the same cosines after one random derangement of the ortholog side (`numpy.random.default_rng(0)`, `:125`, `:130-139`). AUC = ROC-AUC of true-pair cosines against permuted-pair cosines. Input-embedding version for T3a and T3b, with the initial checkpoint's median cosine for reference (`:142-154`); contextual version for T1, T3a, T3b, restricted to pairs where both tokens occur in at least 20 cells (`:168-180`, `:186-195`). No verdict is assigned in code.
   - **V3** (`:220-300`). For each T4 gene, the "family" is every human or mouse token whose gene symbol starts with the same run of at least three letters (`:213-217`, `:233-237`); families with fewer than 3 or more than 600 members are skipped (`:259`). Per gene: ROC-AUC of its cosine to family members against its cosine to 2,000 random tokens (`default_rng(0)`, `:240-250`). Pooled verdict: PASS if mean AUC > 0.65, PARTIAL if > 0.55, else WEAK (`:275`); the cytochrome P450 ("CYP") family is reported separately (`:276-281`).
4. **Per-tier loss**: `eval_mlm.py eval --by-tier` on the C1 probe and masks; each masked token's cross-entropy is assigned to the ortholog tier of that token (`finetune/genecompass/eval_mlm.py:93-99`, `:149-156`).

**Where the V1 labels come from** (`analysis/annotate_celltypes.py`). Per QC sample: `normalize_total(target_sum=1e4)` and `log1p` (`:326-327`); up to 2,000 highly variable genes, flavour `seurat_v3` (`:330-332`); scale with `max_value=10`, PCA with up to 50 components (`:335-337`); neighbours graph with 15 neighbours on 30 components (`:342-343`); Leiden clustering at resolution 0.8 (`:80`, `:344`); score every PanglaoDB cell type that has at least 3 marker genes in the data with `scanpy.tl.score_genes` (`:81`, `:242`); each cluster gets the cell type with the highest mean score, whatever its value (`:263-293`); scores below 0.1 are only flagged for review (`:82`, `:422`). Markers: PanglaoDB file of 27 March 2020, human and mouse entries, mapped to rat by gene symbol (`:77`, `:122`). These are therefore labels derived from the same expression data by per-sample clustering, not author-supplied labels. A second annotation (ScType, `analysis/annotate_sctype.py`) and an LLM-merged consensus (`analysis/consensus_annotations.py`, model `claude-haiku-4-5-20251001`, `:283`) exist in sibling folders but are not what V1 reads.

**Forgetting trajectory** (`analysis/forgetting_analysis.py`; run for April only). Sample 5,000 cells per species with `numpy.random.RandomState(42)` from the human set, the mouse set and the April rat corpus (`:91-107`, `:384-400`, `:808-810`); mask 15% of non-pad positions, replacing every selected token by `<mask>` and setting its value to 0.0 (`:427-477`); score each discovered checkpoint in batches of 64, averaging the batch losses (`:484-550`); also the L2 distance of every encoder layer and of the word embeddings from the base (`:556-630`).

**Annotation recovery** (`analysis/annotation_recovery.py`). Despite its name this is not a test of the model. It rebuilds sample-level metadata (tissue, sex, strain, condition) by joining the Stage 2 preprocessing report, GEO sample tables, file-name patterns and a study-level LLM summary, with precedence GEO > file name > LLM (`:495-513`), and maps tissue strings to MoTrPAC tissues with one call to `claude-haiku-4-5` at temperature 0, cached in `reports/annotations/tissue_motrpac_map.json` (`:94-180`). Its output supplies the tissue hints used by the annotation scripts (`analysis/annotate_sctype.py:24`, `:719`); it is not an input to Stage 7 training or to checks C1-C5.

### Models and software
v2 final save and v2 init (`slurm/analysis/v2_aim1.slurm:24-25`); scikit-learn (`NearestNeighbors`, `KMeans`, `roc_auc_score`, `adjusted_rand_score`), `umap-learn` for the figure (`analyze.py:30-32`, `:93-94`); scanpy for the labels. Installed versions on this machine: scikit-learn 1.8.0, scanpy 1.12, umap-learn 0.5.11 (read for this audit; not recorded with the results).

### Parameters, thresholds and flags
| Name | Value | Where set | What it controls | Rationale given |
|---|---|---|---|---|
| `--target-cells` | 40,000 | `build_inputs.py:157` | size of the V1 subset | none stated |
| `--per-study` | 2,500 | `build_inputs.py:156` | cap per study | "balance the batch-mixing test" (`:156`) |
| `--min-celltype-n` | 50 | `build_inputs.py:158` | rare labels dropped | none stated |
| `--n-t1` | 1,000 | `build_inputs.py:159` | T1 genes tested | none stated |
| `--seed` | 42 | `build_inputs.py:162`, `:164` | study order, row sample, T1 sample | |
| `--n-cells` (human, mouse) | 50,000, a prefix of the dataset, not a random sample | `slurm/analysis/v2_aim1.slurm:38,41`; `extract_embeddings.py:86-87` | context cells | order-preserving cap (`extract_embeddings.py:71`) |
| `--batch-size` | 32 | `extract_embeddings.py:75` | | |
| `--min-cells` | 20 | `extract_embeddings.py:76`; `analyze.py:168` | minimum cells per token for a contextual mean | none stated |
| V1 k | 15 | `analyze.py:57` | neighbours | none stated |
| V1 verdict thresholds | 3 x chance and 0.30; 1.5 x chance | `analyze.py:107-108` | PASS / PARTIAL / WEAK | none stated |
| V1 k-means | k = number of labels, `n_init=4`, `random_state=0` | `analyze.py:88-89` | ARI | |
| V1 UMAP (figure only) | `n_neighbors=30`, `min_dist=0.3`, `random_state=0` | `analyze.py:94` | | |
| V2 / V3 random seed | 0 | `analyze.py:125`, `:240` | derangement; null tokens | |
| V3 family size limits | 3 to 600 | `analyze.py:259` | which symbol roots count | "skip ultra-generic roots (e.g. LOC, ZNF)" (`:259`) |
| V3 null tokens | 2,000 | `analyze.py:241` | | |
| V3 verdict thresholds | 0.65, 0.55 | `analyze.py:275`, `:281` | | none stated |
| Leiden resolution (labels) | 0.8 | `analysis/annotate_celltypes.py:80` | cluster granularity | none stated |
| minimum markers per cell type | 3 | `analysis/annotate_celltypes.py:81` | | |
| forgetting: cells / mask rate / seed / batch | 5,000 / 0.15 / 42 / 64 | `analysis/forgetting_analysis.py:808-811` | | |

### Statistical procedures
None of V1-V3 has a significance test, a confidence interval or a multiplicity correction. V1 reports proportions against a chance baseline. V2 and V3 report ROC-AUCs against a single permutation (V2) or a single random draw of 2,000 tokens (V3). Verdict thresholds are literals in the script; the script was first committed on 2026-06-26 (commit `00c869f`), after April's run of 2026-06-25/26 (`reports/aim1_validation/AIM1_VALIDATION_REPORT.md:2`), and the thresholds are not in any pre-registration. For v2 the procedure, thresholds and seeds were unchanged from April's (`docs/RETRAIN_V2_RESULTS.md:605-607`).

### Checks, validation and their results
V1-V3 (`reports/aim1_validation_v2/metrics.json` for v2; `reports/aim1_validation/metrics.json` for April):

| Check | April | v2 | v2 file:line |
|---|---|---|---|
| V1 cells / cell types / studies | 39,779 / 50 / 17 | 40,332 / 44 / 17 | `metrics.json:3-5` |
| V1 kNN cell-type purity (chance) | 0.813 (0.047) | 0.849 (0.065) | `:8-9` |
| V1 kNN study purity (chance) | 0.975 (0.060) | 0.966 (0.060) | `:10-11` |
| V1 cross-study mixing within cell type | 0.020 | 0.036 | `:12` |
| V1 ARI, k-means vs labels | 0.309 | 0.405 | `:13` |
| V1 verdict | PARTIAL | PARTIAL | `:15` |
| V2 contextual AUC, T1 (pairs scored) | 1.000 (938) | 1.000 (915) | `:38-43` |
| V2 contextual AUC, T3a | 0.973 (597) | 0.948 (591) | `:44-49` |
| V2 contextual AUC, T3b | 0.908 (232) | 0.851 (243) | `:50-55` |
| V2 contextual median cosine, true vs permuted: T1 / T3a / T3b | 0.988 vs 0.712 / 0.900 vs 0.666 / 0.756 vs 0.671 | 0.989 vs 0.715 / 0.885 vs 0.666 / 0.741 vs 0.657 | `:40-41`, `:46-47`, `:52-53` |
| V2 input-embedding AUC, T3a / T3b | 0.998 / 0.991 | 0.962 / 0.875 | `:25`, `:32` |
| V2 input-embedding median cosine to ortholog, init -> final, T3a / T3b | 1.000 -> 0.810 / 1.000 -> 0.801 | 1.000 -> 0.705 / 1.000 -> 0.673 | `:26-27`, `:33-34` |
| V3 pooled mean rank-AUC over 425 T4 genes in 190 families (share with AUC > 0.65) | 0.527 (24.5%) | 0.471 (16.0%) | `:60-64` |
| V3 CYP, 4 genes vs 167 peers | 0.601 | 0.569 | `:95-105` |
| V3 verdict | WEAK | WEAK | `:92`, `:106` |

April's values are from `reports/aim1_validation/metrics.json:3-64`. V1 is not a paired comparison: each model's subset is drawn from its own corpus and the two share 1,935 cells and 6 of 17 studies (`docs/RETRAIN_V2_RESULTS.md:609-612`); C3 is the paired comparison. Reading of V1: neighbourhoods are cell-type-coherent (13 times chance) but almost never mix studies (3.6% of same-type neighbours come from another study; study purity 0.966), i.e. the single-cell embedding is not batch-corrected. Reading of V3: v2's pooled AUC is below 0.5; new rat-only tokens are not nearer their putative family than random tokens.

Per-tier rat gene-ID loss on the C1 probe (nats; `data/results/retrain_v2/mlm_by_tier/*.json`; table in `docs/RETRAIN_V2_RESULTS.md:649-657`):

| Tier (masked tokens) | v2 init | April | v2 after phase 1 | v2 | v2 top-1 |
|---|---|---|---|---|---|
| T1 (235,467) | 4.23 | 3.54 | 3.33 | 3.26 | 21.7% |
| T2a (11,185) | 4.54 | 3.47 | 3.01 | 2.89 | 28.2% |
| T2b (43,842) | 19.97 | 13.77 | 2.58 | 2.41 | 38.4% |
| T3a (11,989) | 4.47 | 3.22 | 2.86 | 2.76 | 29.3% |
| T3b (4,495) | 19.18 | 12.22 | 2.49 | 2.25 | 42.4% |
| T4 (10,641) | 19.20 | 11.00 | 2.50 | 2.20 | 43.0% |
| all (317,619) | 7.13 | 5.31 | 3.16 | 3.06 | 25.5% |

Almost all of the gain over the initial model is on mouse-ortholog and new tokens, and it is already present after phase 1. Descriptive, in-sample, not a criterion (`docs/RETRAIN_V2_RESULTS.md:644-647`).

Forgetting trajectory, April only (`reports/forgetting/forgetting_summary.tsv:2-33`): human / mouse gene-ID loss 4.2451 / 4.1780 for base, 4.3519 / 4.3235 after phase 1, 4.0225 / 4.0230 at `checkpoint-147941`; rat 7.8242 at init -> 3.8956. Total encoder L2 distance from base 0 after phase 1 and 18.68 at the end of phase 2; relative change of the word-embedding table 0.402 after phase 1 and 0.417 at the end (`reports/forgetting/forgetting_report.md:47-56`). No such run exists for v2: the pre-registration lists the id-loss trajectory among the items not rerun, on the grounds that the C1/C2 probe already gives it for base, init and both phases (`docs/RETRAIN_V2_PREREGISTRATION.md:270-271`). These losses are not comparable with the C1/C2 losses (different masking, see caveat 6).

### Deviations, known defects and caveats
1. **"Held-out" is wrong.** The V1 cells are sampled from the training corpus (`build_inputs.py:69`, default and v2 `--corpus` arguments; `slurm/analysis/v2_aim1.slurm:29`), yet the report generator writes "held-out rat cells" and "held-out cell-type clustering" (`analyze.py:99`, `:314`, `:318`; `reports/aim1_validation_v2/AIM1_VALIDATION_REPORT.md:4-6`), and the same wording is in `docs/OPEN_FRONTIER.md:122`, `manuscript/motrpac_genecompass.tex:361` and `:1286`, and `finetune/genecompass/embed_cells.py:7`. `reports/pipeline_report.md:451-452` and `:465` likewise call the April forgetting samples "held-out"; they are drawn from the training sets. There is no held-out set (7d).
2. **V1 labels are automated.** They come from per-sample Leiden clusters labelled by the best-scoring PanglaoDB marker set (above). The label of a cell is a function of its own expression profile and of its sample, so "cell-type purity" partly measures whether the embedding preserves within-sample expression clusters. Label names such as "Nuocytes" or "Myeloid-derived suppressor cells" in a 44-type list (`data/validation/aim1_v2/subset_summary.json`) show the label set is the marker database's vocabulary, not a curated one. The docstring's "consensus cell-type labels" (`build_inputs.py:9`) does not match the folder read (`:39`).
3. **Marker-annotation detail to confirm.** `seurat_v3` highly-variable-gene selection is applied after normalisation and log transform (`analysis/annotate_celltypes.py:326-332`); scanpy documents that flavour as expecting raw counts [confirmed in the installed scanpy 1.12 docstring: "Expects logarithmized data, except when `flavor='seurat_v3'`/`'seurat_v3_paper'`, in which count data is expected"].
4. **V1 uses 17 of 81 studies**, chosen by a seeded shuffle, and joins labels by (study, barcode) although the v2 corpus now carries `sample_id`; 104,128 barcodes were dropped as ambiguous (`logs/v2_aim1_12015439.out`).
5. **V2 coverage and null.** Contextual AUCs are computed only on tokens seen in at least 20 cells on both sides: 591 of 1,351 T3a pairs and 243 of 892 T3b pairs for v2. The human and mouse contexts are the first 50,000 rows of each set. The null is one permutation with no repeat; no interval is reported. The label "PASS" for V2 in `docs/OPEN_FRONTIER.md:118` has no coded or registered pass rule behind it.
6. **`forgetting_analysis.py` does not reproduce training conditions and has stale paths.** (a) Masked values are set to 0.0 (`:460-464`) where the training collator uses -1 (`vendor/GeneCompass/genecompass/data_collator.py:71,77`), and every selected token becomes `<mask>` (no 10% random / 10% unchanged). (b) No attention mask is passed (`:526-532`), so padding positions are attended to. (c) Losses are averaged over batches, not over tokens (`:546-550`). Base human gene-ID loss is 4.245 under this protocol against 3.444 under `eval_mlm.py` (the two also use different cells: 5,000 random cells here, the 2,000 probe cells there). (d) The species-embedding drift is looked up under `bert.embeddings.cls_embedding.weight` (`:609`), but the tensor is `bert.cls_embedding.weight` (`finetune/genecompass/rat_model_init.py:369`), so that column is "N/A" in every row of the report (`reports/forgetting/forgetting_report.md:47-56`). (e) The rat corpus path is the April corpus (`:103`) although the default model paths now point at v2 (`:111-112`), and the project root is an absolute literal (`:66`). (f) Its docstring points to `analysis/aim1_validation/analyze.py` for C1/C2 (`:39-40`); C1/C2 are in `finetune/genecompass/eval_mlm.py` and `score_prereg_c1c2.py`.
7. **Stale docstring in `analyze.py`.** The header describes V1 as silhouette by cell type vs by study (`analyze.py:5-6`); the code computes kNN purities and never calls `silhouette_score` (imported at `:30`). A comment in the function explains the choice ("NOT global silhouette, which is confounded", `:58`), which indicates the metric was changed after the header was written; the header was not updated.
8. **`annotation_recovery.py` and `consensus_annotations.py` depend on LLM calls** (`analysis/annotation_recovery.py:147-152`; `analysis/consensus_annotations.py:283`) whose outputs are cached; a rebuild without the cache is not guaranteed to return the same mapping. `annotation_recovery.py` counts cells from the April corpus (`:720`).
9. **Records are on disk only.** `reports/` is gitignored; `analysis/README.md` is a single heading (`analysis/README.md:1`).
10. **No test of generalisation to an unseen rat study exists anywhere in Stage 7.**

### Methods references
- GeneCompass (the homolog-similarity analysis is described in the report text as "the GeneCompass Fig-2a analogue", `analyze.py:343`): Yang et al., Cell Research 2024;34:830-845 (`manuscript/motrpac_genecompass.tex:1487-1489`).
- ScType: "Ianevski et al., Nat Commun 2022" (as given in `analysis/annotate_sctype.py:5`).
- PanglaoDB marker database (file `PanglaoDB_markers_27_Mar_2020.tsv.gz`, `analysis/annotate_celltypes.py:77`): Franzen, Gan, Bjorkegren 2019, Database [verified against Crossref; see REFERENCES.md].
- Leiden clustering: Traag, Waltman, van Eck 2019, Sci Rep [verified against Crossref; see REFERENCES.md]. scanpy: Wolf, Angerer, Theis 2018, Genome Biol [verified against Crossref; see REFERENCES.md]. UMAP: McInnes, Healy, Melville 2018, arXiv:1802.03426 [verified against arXiv 1802.03426; see REFERENCES.md].
- Adjusted Rand index: Hubert and Arabie 1985, J Classif [verified against Crossref; see REFERENCES.md].
- LLM used for tissue mapping and label consensus: Anthropic Claude Haiku 4.5 (model strings in `analysis/annotation_recovery.py:148` and `analysis/consensus_annotations.py:283`).

---

## Open items for this section (could not be determined from the repo)
- Which upstream download (the "0.5M" or the "5M" row of `vendor/GeneCompass/README.md:52-53`) produced `randsel_500w_human` / `randsel_500w_mouse`; the files hold about 5.0M and 4.97M cells.
- The exact row count of April's training mix and the contents of April's `sorted_length.pickle` (not opened: a 33-million-entry pickle).
- The base model's parameter count as logged by any tool (only the arithmetic value 132,601,217 is available).
- Why the untrained model's loss on mouse-ortholog tokens in rat cells (about 19-20 nats) is above the uniform-guess level (10.92 nats).
- How many optimisation steps had non-finite gradients (only every 50th step's gradient norm is logged), and what the optimiser did on those steps (HF / PyTorch internals, not read for this audit).
- The SLURM log of the first phase-2 attempt (overwritten on requeue).

