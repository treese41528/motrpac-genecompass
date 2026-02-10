import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import shutil
from downstream_tasks.grn_inference.EmbeddingEvaluator import DeepSEM_result

# Base paths
base_path = os.path.abspath("/workspace/GeneCompass")
sys.path.append(base_path)

# Copy required files to current directory
grn_inference_path = os.path.join(base_path, "downstream_tasks/grn_inference")
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

# Rest of the paths
deepsem_path = os.path.join(grn_inference_path, "DeepSEM-master")
beeline_path = os.path.join(deepsem_path, "BEELINE-data/inputs/scRNA-Seq/hESC")
deepsem_result_path = os.path.join(deepsem_path, "result/GeneCompass_cls_new-emb-0708")

expression_data_path = os.path.join(beeline_path, "BL-hESC-ChIP-seq-no-peca-ExpressionData.csv")
network_path = os.path.join(beeline_path, "BL-hESC-ChIP-seq-no-peca-network.csv")

model_path = os.path.join(base_path, "pretrained_models/GeneCompass_Base")
dataset_path = os.path.join(base_path, "data/genecompass_0.05M/randsel_5w_human")
prior_knowledge_path = os.path.join(base_path, "prior_knowledge")
embedding_file_path = os.path.join(grn_inference_path, "GeneCompass-emb-test.npy")

# Matrix paths
similarity_matrix_path = os.path.join(grn_inference_path, "GeneCompass-similarity-matrix.npy")
gene_names_path = os.path.join(grn_inference_path, "GeneCompass-gene-names.npy")

# Load matrices
similarity_matrix = np.load(similarity_matrix_path)
gene_names = np.load(gene_names_path, allow_pickle=True)

print(f"Similarity matrix shape: {similarity_matrix.shape}")
print(f"Gene names count: {len(gene_names)}")

# Create result directory
os.makedirs(deepsem_result_path, exist_ok=True)
if os.path.exists(embedding_file_path):
    print(f"Precomputed embeddings found at {embedding_file_path}")
else:
    print(f"Precomputed embeddings NOT found at {embedding_file_path}")

print("Running DeepSEM using EmbeddingEvaluator...")
try:
    DeepSEM_result(
        path=base_path,
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
except Exception as e:
    print(f"Error running DeepSEM: {str(e)}")

# Process results 
result_files = [f for f in os.listdir(deepsem_result_path) 
                if f.startswith("GRN_inference_result_")]

if result_files:
    def get_result_num(filename):
        try:
            return int(filename.split("_")[-1].split(".")[0])
        except (IndexError, ValueError):
            return -1
            
    latest_result = sorted(result_files, key=get_result_num)[-1]
    result_file_path = os.path.join(deepsem_result_path, latest_result)
    
    print(f"Loading GRN results from: {result_file_path}")
    grn_results = pd.read_csv(result_file_path, sep="\t")
    print(f"GRN Results: {grn_results.head()}")
    
    # Visualize top interactions
    print("Visualizing Top GRN Interactions...")
    top_interactions = grn_results.sort_values(by="EdgeWeight", ascending=False).head(10)
    
    G = nx.DiGraph()
    for _, row in top_interactions.iterrows():
        G.add_edge(row["Gene1"], row["Gene2"], weight=row["EdgeWeight"])
        
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue",
            edge_color="gray", node_size=2000, font_size=10)
    plt.title("Top Gene Regulatory Interactions")
    
    network_save_path = os.path.join(base_path, "network.png")
    plt.savefig(network_save_path, dpi=300)
    print(f"Network Graph saved to: {network_save_path}")
else:
    print("No GRN inference results found")

# Clean up copied files
for file in files_to_copy:
    if os.path.exists(f"./{file}"):
        os.remove(f"./{file}")