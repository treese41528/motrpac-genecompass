import os
import torch
import numpy as np
import pickle
import shutil

from datasets import load_from_disk
from EmbeddingGenerator import get_GeneCompass_cls_new_embedding
from downstream_tasks.grn_inference.EmbeddingEvaluator import DeepSEM_result

# Force synchronous CUDA operations
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Initialize CUDA
if torch.cuda.is_available():
    torch.cuda.init()
    # Force torch to reinitialize CUDA
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"Using GPU: {torch.cuda.get_device_name()}")

path = "/workspace/GeneCompass/"

# dataset_path = f"{path}/data/MoTrPAC/MoTrPAC_Top2048_ABS_Condition_geneCompass/MoTrPAC_Top2048_ABS_Condition_processed"
dataset_path = f"{path}/data/MoTrPAC/MoTrPAC_data_Top2048_VAR_Global_geneCompass/processed"
# Load dataset
data = load_from_disk(dataset_path)
# Debug first sample's data
print("First sample debugging:")
sample = data[0]
print(f"Max input_id value: {max(sample['input_ids'])}")
print(f"Min input_id value: {min(sample['input_ids'])}")
print(f"Unique input_ids: {len(set(sample['input_ids']))}")
# Compare with working dataset
working_data = load_from_disk(f"{path}/data/genecompass_0.05M/randsel_5w_human")
working_sample = working_data[0]
print("\nWorking sample debugging:")
print(f"Max input_id value: {max(working_sample['input_ids'])}")
print(f"Min input_id value: {min(working_sample['input_ids'])}")
print(f"Unique input_ids: {len(set(working_sample['input_ids']))}")





def fetch_embeddings(dataset_path, embeddings_save_path, similarity_save_path, gene_names_save_path):
    # Also check the token mappings
    grn_genes_path = f"{path}/downstream_tasks/grn_inference/GRNgene.pkl"
    id_token_path = f"{path}/prior_knowledge/h&m_token1000W.pickle"
    gene_dict_path = f"{path}/downstream_tasks/grn_inference/Gene_id_name_dict.pickle"
    with open(id_token_path, 'rb') as f:
        id_token = pickle.load(f)
    print(f"\nToken map size: {len(id_token)}")
    print(f"Max token value: {max(id_token.values())}")
    prior_embedding_path = f"{path}/prior_knowledge/"
    checkpoint_path = f"{path}/pretrained_models/GeneCompass_Base"
    token_map_path = f"{path}/prior_knowledge/human_mouse_tokens.pickle"
    gene_id_name_path = f"{path}/downstream_tasks/grn_inference/Gene_id_name_dict.pickle"
    promoter_path = f"{path}/prior_knowledge/promoter_emb/human_emb_768.pickle"
    homologous_map_path = f"{path}/prior_knowledge/homologous_hm_token.pickle"
    grn_path = f"{path}/downstream_tasks/grn_inference/GRNgene.pkl"
    # Save paths

    # Ensure directory exists
    os.makedirs(os.path.dirname(embeddings_save_path), exist_ok=True)
    
    gene_names, similarity_matrix, final_embeddings = get_GeneCompass_cls_new_embedding(
    path=path,
    dataset_path=dataset_path,
    checkpoint_path=checkpoint_path,
    prior_embedding_path=prior_embedding_path,
    get_emb=True,
    emb_file_path=embeddings_save_path
    )


    print("\nEmbedding generation complete.")
    print(f"Gene Names count: {len(gene_names)}")
    print(f"Similarity Matrix shape: {similarity_matrix.shape}")
    print(f"Final Embeddings shape: {final_embeddings.shape}")

    # Save results
    np.save(similarity_save_path, similarity_matrix)
    np.save(gene_names_save_path, np.array(gene_names, dtype=object))
    print(f"\nResults saved to: {os.path.dirname(embeddings_save_path)}")

    return gene_names, similarity_matrix, final_embeddings


def fetch_GRN(deepsem_result_path, embedding_file_path):
    # Copy required files to current directory
    grn_inference_path = os.path.join(path, "downstream_tasks/grn_inference")
    files_to_copy = [
        "GRNgene.pkl",
        "h_m_token2000W.pickle",
        "Gene_id_name_dict.pickle"
    ]

    for file in files_to_copy:
        src = os.path.join(grn_inference_path, file)
        dst = f"./{file}"
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {file} to current directory")

    deepsem_path = os.path.join(grn_inference_path, "DeepSEM-master")
    beeline_path = os.path.join(deepsem_path, "BEELINE-data/inputs/scRNA-Seq/hESC")
    expression_data_path = os.path.join(beeline_path, "BL-hESC-ChIP-seq-no-peca-ExpressionData.csv")
    network_path = os.path.join(beeline_path, "BL-hESC-ChIP-seq-no-peca-network.csv")

    prior_knowledge_path = deepsem_path
    model_path = os.path.join(path, "pretrained_models/GeneCompass_Base")

    DeepSEM_result(
        path=path,
        data_file=expression_data_path,
        net_file=network_path,
        save_name=deepsem_result_path,
        model="GeneCompass_cls_new",
        dataset_path=dataset_path,
        checkpoint_path=model_path,
        n_epochs=40,
        get_emb=False,
        emb_file_path=embedding_file_path,
        prior_embedding_path=prior_knowledge_path
    )



def main():
    # Global
    embeddings_save_path = f"{path}/Embeddings/MoTrPAC_Top2048_VAR_Global_geneCompass/GeneCompass-emb-MoTrPAC_Top2048_VAR_Global.npy"
    similarity_save_path = f"{path}/Embeddings/MoTrPAC_Top2048_VAR_Global_geneCompass/GeneCompass-similarity-matrix-MoTrPAC_Top2048_VAR_Global.npy"
    gene_names_save_path = f"{path}/Embeddings/MoTrPAC_Top2048_VAR_Global_geneCompass/GeneCompass-gene-names-MoTrPAC_Top2048_VAR_Global.npy"
    dataset_path = f"{path}/data/MoTrPAC/MoTrPAC_data_Top2048_VAR_Global_geneCompass/processed"


    gene_names, similarity_matrix, final_embeddings = fetch_embeddings(dataset_path, embeddings_save_path, similarity_save_path, gene_names_save_path)
    
    
    deepsem_result_path = f"{path}/Embeddings/MoTrPAC_Top2048_VAR_Global_geneCompass/GRN"
    fetch_GRN(deepsem_result_path, embeddings_save_path)


    # Condition 
    embeddings_save_path = f"{path}/Embeddings/MoTrPAC_data_Top2048_ABS_Condition_geneCompass/GeneCompass-emb-MoTrPAC_Top2048_ABS_Condition.npy"
    similarity_save_path = f"{path}/Embeddings/MoTrPAC_data_Top2048_ABS_Condition_geneCompass/GeneCompass-similarity-matrix-MoTrPAC_Top2048_ABS_Condition.npy"
    gene_names_save_path = f"{path}/Embeddings/MoTrPAC_data_Top2048_ABS_Condition_geneCompass/GeneCompass-gene-names-MoTrPAC_Top2048_ABS_Condition.npy"
    dataset_path = f"{path}/data/MoTrPAC/MoTrPAC_data_Top2048_ABS_Condition_geneCompass/processed"


    gene_names, similarity_matrix, final_embeddings = fetch_embeddings(dataset_path, embeddings_save_path, similarity_save_path, gene_names_save_path)

    
    deepsem_result_path = f"{path}/Embeddings/MoTrPAC_data_Top2048_ABS_Condition_geneCompass/GRN"
    fetch_GRN(deepsem_result_path, embeddings_save_path)



    # Tissue
    # embeddings_save_path = f"{path}/Embeddings/MoTrPAC_data_Top2048_VAR_Tissue_geneCompass/GeneCompass-emb-MoTrPAC_Top2048_VAR_Tissue.npy"
    # similarity_save_path = f"{path}/Embeddings/MoTrPAC_data_Top2048_VAR_Tissue_geneCompass/GeneCompass-similarity-matrix-MoTrPAC_Top2048_VAR_Tissue.npy"
    # gene_names_save_path = f"{path}/Embeddings/MoTrPAC_data_Top2048_VAR_Tissue_geneCompass/GeneCompass-gene-names-MoTrPAC_Top2048_VAR_Tissue.npy"
    # dataset_path = f"{path}/data/MoTrPAC/MoTrPAC_data_Top2048_VAR_Tissue_geneCompass/processed"


    # gene_names, similarity_matrix, final_embeddings = fetch_embeddings(dataset_path, embeddings_save_path, similarity_save_path, gene_names_save_path)




if __name__ == "__main__":
    main()







