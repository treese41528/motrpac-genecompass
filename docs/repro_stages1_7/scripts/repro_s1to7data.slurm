#!/bin/bash
#SBATCH --job-name=repro_s1to7data
#SBATCH --account=reese18
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/repro_%x_%j.out
# Reproducibility check 2026-10-31: Stage 1 input check; Stage 3; Stage 6 (family, GRN); Stage 7 step 1 (init) +
# token dictionary; Stage 4 (16 scatter tasks, as recorded) + gather + hybrid medians; Stage 5 (50 tasks, as recorded)
# + assemble; training mix; eval probe. All outputs under scratch; comparisons against the canonical outputs.
S=/scratch/gilbreth/reese18/motrpac_repro_20261031
R=/depot/reese18/apps/motrpac-genecompass
cd $R
source /depot/reese18/apps/motrpac-env/bin/activate
export TMPDIR=$S/tmp TEMP=$S/tmp TMP=$S/tmp HF_HOME=$S/tmp/hf HF_DATASETS_CACHE=$S/tmp/hf_ds XDG_CACHE_HOME=$S/tmp/xdg \
       MPLCONFIGDIR=$S/tmp/mpl NUMBA_CACHE_DIR=$S/tmp/numba
export PIPELINE_PROFILE=$S/profiles/repro_s3to6.yaml
mkdir -p $S/cmp $S/logs
T=$S/cmp/timings_s1to7data.tsv; echo -e "step\texit\tseconds" > $T
run() { local name=$1; shift; local t0=$(date +%s); echo "=== $name $(date)"; "$@" > $S/logs/$name.log 2>&1; local rc=$?; echo -e "$name\t$rc\t$(( $(date +%s) - t0 ))" | tee -a $T; tail -5 $S/logs/$name.log; return $rc; }
par() { local name=$1 n=$2 P=$3; shift 3; local t0=$(date +%s); echo "=== $name $(date)";
  seq 0 $((n-1)) | xargs -P $P -I{} bash -c "$* --task-id {} --n-tasks $n > $S/logs/${name}_{}.log 2>&1 || echo FAIL {}" > $S/logs/${name}_fails.txt
  echo -e "$name\t$(wc -l < $S/logs/${name}_fails.txt)fails\t$(( $(date +%s) - t0 ))" | tee -a $T; }

# Stage 1: inputs v2 used
run stage1_verify python $S/scripts/verify_stage1.py $S/cmp/stage1_qc_inputs.tsv 16
md5sum data/references/biomart/rat_gene_info.tsv data/references/biomart/rat_human_orthologs.tsv data/references/biomart/rat_mouse_orthologs.tsv \
  data/references/biomart/mouse_human_orthologs.tsv data/references/hgnc/hgnc_complete_set.txt vendor/GeneCompass/prior_knowledge/human_mouse_tokens.pickle \
  vendor/GeneCompass/prior_knowledge/homologous_hm_token.pickle data/training/gene_universe/pruned_gene_universe.tsv \
  vendor/GeneCompass/scdata/dict/human_gene_median_after_filter.pickle > $S/cmp/reference_input_md5.txt

# Stage 3 (+ token dictionary, which Stage 7 reads)
run stage3 python pipeline/run_stage3.py
run token_dict python finetune/genecompass/build_rat_token_dictionary.py --rat-tokens $S/s3/ortholog_mappings/rat_tokens.pickle \
    --output $S/s3/ortholog_mappings/rat_human_mouse_tokens.pickle
run cmp_stage3 python $S/scripts/cmp_dir.py $S/s3/ortholog_mappings $R/data/training/ortholog_mappings $S/cmp/stage3_ortholog_mappings.tsv 4

# Stage 6: orchestrator dry-run (records the step list), family + GRN (the priors v2's init reads for T4)
run stage6_dryrun python pipeline/run_stage6.py --dry-run --from 2
run stage6_family python pipeline/06_prior_knowledge/build_family_embedding.py
run stage6_grn python pipeline/06_prior_knowledge/build_grn_transfer.py
run cmp_stage6 python $S/scripts/cmp_dir.py $S/s6/prior_knowledge $R/data/training/prior_knowledge $S/cmp/stage6_prior_knowledge.tsv 4

# Stage 7 step 1: init from the reproduced token dict + reproduced family/GRN (coexp/promoter: canonical files, zeroed for T4 by policy)
mkdir -p $S/s6/init_stage6
ln -sf $S/s6/prior_knowledge/family_embeddings.pkl $S/s6/prior_knowledge/grn_embeddings.pkl $S/s6/init_stage6/
ln -sf $R/data/training/prior_knowledge/coexp_embeddings.pkl $R/data/training/prior_knowledge/promoter_embeddings.pkl $S/s6/init_stage6/
run init python finetune/genecompass/rat_model_init.py --checkpoint $R/vendor/GeneCompass/pretrained_models/GeneCompass_Base \
    --rat-token-dict $S/s3/ortholog_mappings/rat_human_mouse_tokens.pickle --stage6-dir $S/s6/init_stage6 \
    --homologous-gene-path $R/vendor/GeneCompass/prior_knowledge/homologous_hm_token.pickle --output-dir $S/s7/init_v2_repro \
    --t4-init random --t4-std 0.02 --species-noise-std 0.02 --coexp-rescale 3.9 --t3-priors ortholog --t4-zero-priors promoter,coexp
run cmp_init python $S/scripts/cmp_init.py $S/s7/init_v2_repro/pytorch_model.bin $R/data/models/rat_genecompass_init_v2/pytorch_model.bin \
    $S/cmp/stage7_init_tensors.tsv $S/s3/ortholog_mappings/rat_token_mapping.tsv
run init2 python finetune/genecompass/rat_model_init.py --checkpoint $R/vendor/GeneCompass/pretrained_models/GeneCompass_Base \
    --rat-token-dict $S/s3/ortholog_mappings/rat_human_mouse_tokens.pickle --stage6-dir $S/s6/init_stage6 \
    --homologous-gene-path $R/vendor/GeneCompass/prior_knowledge/homologous_hm_token.pickle --output-dir $S/s7/init_v2_repro2 \
    --t4-init random --t4-std 0.02 --species-noise-std 0.02 --coexp-rescale 3.9 --t3-priors ortholog --t4-zero-priors promoter,coexp
run cmp_init_self python $S/scripts/cmp_init.py $S/s7/init_v2_repro/pytorch_model.bin $S/s7/init_v2_repro2/pytorch_model.bin \
    $S/cmp/stage7_init_repro_vs_repro2.tsv $S/s3/ortholog_mappings/rat_token_mapping.tsv
rm -rf $S/s7/init_v2_repro2

# Stage 4: 16 scatter tasks (stage4_manifest.json n_scatter_tasks=16), gather, hybrid medians
par stage4_scatter 16 16 python pipeline/04_gene_medians/compute_gene_medians.py
run stage4_gather python pipeline/04_gene_medians/gather_gene_medians.py
run stage5_hybrid python pipeline/05_tokenization/build_hybrid_medians.py
run cmp_stage4 python $S/scripts/cmp_dir.py $S/s4/gene_medians_v2 $R/data/training/gene_medians_v2 $S/cmp/stage4_gene_medians.tsv 8
run cmp_stage4_scatter python $S/scripts/cmp_dir.py $S/s4/gene_medians_v2/scatter $R/data/training/gene_medians_v2/scatter $S/cmp/stage4_scatter.tsv 16

# Stage 5: 50 tokenize tasks (stage5_manifest.json n_scatter_tasks=50), assemble
par stage5_tokenize 50 16 python pipeline/05_tokenization/tokenize_corpus.py
run stage5_assemble python pipeline/05_tokenization/assemble_corpus.py --expected-tasks 50
run cmp_stage5_top python $S/scripts/cmp_dir.py $S/s5/tokenized_corpus_v2 $R/data/training/tokenized_corpus_v2 $S/cmp/stage5_top.tsv 2
run cmp_stage5_manifests python $S/scripts/cmp_dir.py $S/s5/tokenized_corpus_v2/manifests $R/data/training/tokenized_corpus_v2/manifests $S/cmp/stage5_manifests.tsv 8
run cmp_stage5_shards python $S/scripts/cmp_dir.py $S/s5/tokenized_corpus_v2/shards $R/data/training/tokenized_corpus_v2/shards $S/cmp/stage5_shards.tsv 16
run cmp_stage5_dataset python $S/scripts/cmp_dir.py $S/s5/tokenized_corpus_v2/dataset $R/data/training/tokenized_corpus_v2/dataset $S/cmp/stage5_dataset.tsv 16
du -sh $S/s5 >> $T

# Training mix (from the reproduced corpus; human/mouse through symlinked dirs so nothing is written beside the vendor data)
mkdir -p $S/s7/src
for sp in human mouse; do d=$S/s7/src/randsel_500w_$sp; mkdir -p $d; ln -sf $R/vendor/GeneCompass/data/randsel_500w_$sp/* $d/; done
run mix python finetune/genecompass/build_mixed_species_dataset.py --rat-dataset $S/s5/tokenized_corpus_v2/dataset \
    --human-dataset $S/s7/src/randsel_500w_human --mouse-dataset $S/s7/src/randsel_500w_mouse --output $S/s7/mixed_species_dataset_v2
run cmp_mix python $S/scripts/cmp_dir.py $S/s7/mixed_species_dataset_v2 $R/data/training/mixed_species_dataset_v2 $S/cmp/stage7_mix.tsv 16

# Eval probe (pre-registration §5/§6), from the reproduced mix
run probe python finetune/genecompass/eval_mlm.py build-probe --mix $S/s7/mixed_species_dataset_v2 --out $S/s7/eval_probe_v2
(cd $S/s7/eval_probe_v2 && md5sum data-00000-of-00001.arrow dataset_info.json state.json) > $S/cmp/stage7_probe_md5_repro.txt
diff <(sort -k2 $S/cmp/stage7_probe_md5_repro.txt) <(sort -k2 $R/data/training/eval_probe_v2.md5) > $S/cmp/stage7_probe_md5_diff.txt; echo "probe md5 diff rc=$?" | tee -a $T
echo "ALL DONE $(date)"
