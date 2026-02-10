import os
import json
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import sys
import requests
from xml.etree.ElementTree import fromstring
from io import StringIO

from datasets import Dataset, Features, Sequence, Value

def explore_pickle(file_path, print_values=False):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"\n--- Exploring {file_path} ---")
        if isinstance(data, dict):
            print(f"Type: Dictionary | Keys: {list(data.keys())[:10]} | Total Keys: {len(data)}")
            first_value = list(data.values())[0]
            if isinstance(first_value, (list, tuple, dict, str)):
                print(f"Sample Value Length: {len(first_value)}")
            else:
                print(f"Sample Value: {first_value}")
            if print_values:
                print(f"Sample Values: {list(data.values())[:10]}")
            if all(isinstance(k, str) and isinstance(v, int) for k, v in data.items()):
                print("This appears to be a token dictionary.")
        elif isinstance(data, list):
            print(f"Type: List | First 5 Elements: {data[:5]} | Total Elements: {len(data)}")
        elif isinstance(data, set):
            print(f"Type: Set | First 5 Elements: {list(data)[:5]} | Total Elements: {len(data)}")
        else:
            print(f"Type: {type(data)} | Preview: {str(data)[:100]}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

def strip_version(ensembl_id):
    """Strip version number from Ensembl ID if present."""
    return ensembl_id.split('.')[0].strip().upper()

def split_by_groups(raw_dataset_path, output_dir, group_columns):
    """
    Splits the raw dataset into subsets based on unique values in specified group columns.
    """
    print("Loading raw dataset...")
    raw_df = pd.read_csv(raw_dataset_path)

    os.makedirs(output_dir, exist_ok=True)
    for col in group_columns:
        if col not in raw_df.columns:
            print(f"Column '{col}' not found. Skipping...")
            continue
        
        print(f"Splitting dataset by '{col}'...")
        for value in raw_df[col].unique():
            subset_df = raw_df[raw_df[col] == value]
            if subset_df.empty:
                print(f"No data for '{col}={value}'. Skipping...")
                continue
            
            sanitized_value = str(value).replace(" ", "_").replace("/", "_").replace("\\", "_")
            subset_path = os.path.join(output_dir, f"{col}_{sanitized_value}_raw.csv")
            subset_df.to_csv(subset_path, index=False)
            print(f"Saved subset: {subset_path}")



def create_expression_matrix_from_raw(dataset_path, gene_map_path, output_path):
    """Creates a gene x sample expression matrix from the full raw dataset."""

    print(f"Loading raw dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    print("Loading gene map...")
    with open(gene_map_path, "rb") as f:
        gene_map = pickle.load(f)

    print(f"Dataset dimensions: {df.shape}")
    samples, expressions = [], []
    unique_genes = set()  # Track unique genes across all samples
    print("Processing all samples to generate a unified expression matrix...")

    # Process each sample
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        ensembl_ids = [strip_version(eid) for eid in row['feature_ids_human'].split(";")]
        expression_values = list(map(float, row['normalized_values'].split(";")))

        # Map Ensembl IDs to Gene Symbols and aggregate values
        gene_expr = {}
        gene_counts = {}
        for eid, value in zip(ensembl_ids, expression_values):
            gene_symbol = gene_map.get(eid, None)
            if gene_symbol:
                unique_genes.add(gene_symbol)  # Add to the global unique genes set
                if gene_symbol not in gene_expr:
                    gene_expr[gene_symbol] = value
                    gene_counts[gene_symbol] = 1
                else:
                    gene_expr[gene_symbol] += value
                    gene_counts[gene_symbol] += 1

        # Average duplicate gene values
        gene_expr = {gene: gene_expr[gene] / gene_counts[gene] for gene in gene_expr}

        samples.append(row['sample_id'])
        expressions.append(gene_expr)

    print(f"Number of samples processed: {len(samples)}")
    print(f"Total unique genes across all samples: {len(unique_genes)}")

    # Construct matrix using the unique gene list
    all_genes = sorted(unique_genes)
    matrix_data = [[expr.get(gene, 0.0) for gene in all_genes] for expr in expressions]
    expression_matrix = pd.DataFrame(matrix_data, index=samples, columns=all_genes)

    print(f"Expression matrix dimensions: {expression_matrix.shape}")
    expression_matrix.to_csv(output_path)
    print(f"Unified expression matrix saved to {output_path} with shape {expression_matrix.shape}")

    return expression_matrix


def create_expression_matrix_from_subset(subset_path, gene_map_path, output_path):
    """Create a unique expression matrix for a given subset."""
    print(f"Loading subset from: {subset_path}")
    df = pd.read_csv(subset_path)

    print("Loading gene map...")
    with open(gene_map_path, "rb") as f:
        gene_map = pickle.load(f)

    print("Processing subset to generate an expression matrix...")
    samples, expressions = [], []
    unique_genes = set()

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        ensembl_ids = [strip_version(eid) for eid in row['feature_ids_human'].split(";")]
        expression_values = list(map(float, row['normalized_values'].split(";")))

        gene_expr = {}
        gene_counts = {}
        for eid, value in zip(ensembl_ids, expression_values):
            gene_symbol = gene_map.get(eid, None)
            if gene_symbol:
                unique_genes.add(gene_symbol)
                if gene_symbol not in gene_expr:
                    gene_expr[gene_symbol] = value
                    gene_counts[gene_symbol] = 1
                else:
                    gene_expr[gene_symbol] += value
                    gene_counts[gene_symbol] += 1

        # Average duplicate genes
        gene_expr = {gene: gene_expr[gene] / gene_counts[gene] for gene in gene_expr}

        samples.append(row['sample_id'])
        expressions.append(gene_expr)

    print(f"Number of unique genes in subset: {len(unique_genes)}")

    all_genes = sorted(unique_genes)
    matrix_data = [[expr.get(gene, 0.0) for gene in all_genes] for expr in expressions]
    expression_matrix = pd.DataFrame(matrix_data, index=samples, columns=all_genes)

    print(f"Expression matrix dimensions: {expression_matrix.shape}")
    expression_matrix.to_csv(output_path)
    print(f"Expression matrix saved to: {output_path}")
    return expression_matrix




# BioMart Client for Transcription Factors
class BiomartTFClient:
    def __init__(self):
        self.server = "https://www.ensembl.org/biomart/martservice"

    def get_transcription_factors(self):
        print("Querying BioMart for transcription factors...")
        query = """<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE Query>
        <Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">
            <Dataset name="hsapiens_gene_ensembl" interface="default">
                <Filter name="go" value="GO:0003700"/> <!-- Transcription factors -->
                <Attribute name="ensembl_gene_id"/>
                <Attribute name="external_gene_name"/>
            </Dataset>
        </Query>"""
        
        try:
            # Use GET request with the query parameter
            response = requests.get(
                self.server, 
                params={'query': query.strip()},  # Pass the query as a parameter
                timeout=120
            )
            response.raise_for_status()

            # Check if the response is in the expected TSV format
            if response.headers.get("Content-Type", "").startswith("text/plain"):
                print("BioMart query successful.")
                return pd.read_csv(StringIO(response.text), sep="\t", header=0)
            else:
                print(f"Unexpected content type: {response.headers.get('Content-Type', '')}")
                print(response.text)  # Print HTML content for debugging
        except requests.exceptions.RequestException as e:
            print(f"Failed to query BioMart: {e}")
        return None



# Load or Fetch Transcription Factor List
def load_or_fetch_tf_list(cache_path="tf_list.csv"):
    if os.path.exists(cache_path):
        print(f"Loading cached transcription factor list from {cache_path}")
        tf_list = pd.read_csv(cache_path)

        # Normalize column names
        tf_list.rename(
            columns={
                'Gene stable ID': 'ensembl_gene_id',
                'Gene name': 'external_gene_name'
            },
            inplace=True
        )
        print("Normalized columns in TF list DataFrame:", tf_list.columns.tolist())
        return tf_list
    else:
        client = BiomartTFClient()
        tf_list = client.get_transcription_factors()
        if tf_list is not None:
            # Normalize and save the DataFrame
            tf_list.rename(
                columns={
                    'Gene stable ID': 'ensembl_gene_id',
                    'Gene name': 'external_gene_name'
                },
                inplace=True
            )
            tf_list.to_csv(cache_path, index=False)
            return tf_list
        else:
            print("Failed to retrieve transcription factors.")
            return None



# Create Unified TF-Gene Interaction Network
def create_unified_network(data_path, gene_map_path, tf_list, output_path, top_n=80):
    print(f"Loading full dataset: {data_path}")
    full_df = pd.read_csv(data_path, index_col=0)

    print("Loading gene map...")
    with open(gene_map_path, "rb") as f:
        gene_map = pickle.load(f)

    print("Filtering transcription factors...")
    tf_ids = set(tf_list['ensembl_gene_id'])  # Extract TFs from tf_list
    print(f"Number of transcription factors (Ensembl IDs): {len(tf_ids)}")

    networks = []

    print("Processing transcription factors and genes...")
    for idx, row in tqdm(full_df.iterrows(), total=full_df.shape[0]):
        feature_ids = row['feature_ids_human'].split(";")
        normalized_values = list(map(float, row['normalized_values'].split(";")))
        gene_values = dict(zip(feature_ids, normalized_values))

        for tf_id in tf_ids:
            if tf_id in gene_values:
                # Sort genes by expression value and select top N
                sorted_genes = sorted(gene_values.items(), key=lambda x: x[1], reverse=True)
                top_genes = sorted_genes[:top_n]
                for gene_id, value in top_genes:
                    if gene_id != tf_id:
                        tf_name = gene_map.get(tf_id)
                        gene_name = gene_map.get(gene_id)
                        
                        # Skip if no mapping is found for either TF or gene
                        if tf_name and gene_name:
                            networks.append({'Gene1': tf_name, 'Gene2': gene_name})

    print(f"Total interactions: {len(networks)}")
    if networks:
        network_df = pd.DataFrame(networks)
        print("Dropping duplicate gene pairs...")
        network_df['GenePair'] = network_df['Gene1'] + "_" + network_df['Gene2']
        network_df = network_df.drop_duplicates(subset=['GenePair']).drop(columns=['GenePair'])
        print(f"Total interactions after deduplication: {network_df.shape[0]}")
    else:
        print("No interactions found. Creating an empty DataFrame.")
        network_df = pd.DataFrame(columns=['Gene1', 'Gene2'])

    print(f"Saving unified network to: {output_path}")
    network_df.to_csv(output_path, index=False)



def create_network_for_subset(subset_path, gene_map_path, tf_list, output_path, top_n=100):
    """Create a unique regulatory network for a given subset."""
    print(f"Loading subset from: {subset_path}")
    df = pd.read_csv(subset_path)

    print("Loading gene map...")
    with open(gene_map_path, "rb") as f:
        gene_map = pickle.load(f)

    tf_ids = set(tf_list['ensembl_gene_id'])
    print(f"Number of transcription factors (TFs): {len(tf_ids)}")

    networks = []

    print("Processing subset to generate a regulatory network...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        feature_ids = row['feature_ids_human'].split(";")
        normalized_values = list(map(float, row['normalized_values'].split(";")))
        gene_values = dict(zip(feature_ids, normalized_values))

        for tf_id in tf_ids:
            if tf_id in gene_values:
                sorted_genes = sorted(gene_values.items(), key=lambda x: x[1], reverse=True)[:top_n]
                for gene_id, value in sorted_genes:
                    if gene_id != tf_id:
                        tf_name = gene_map.get(tf_id)
                        gene_name = gene_map.get(gene_id)
                        if tf_name and gene_name:
                            networks.append({'Gene1': tf_name, 'Gene2': gene_name})

    print(f"Total interactions: {len(networks)}")
    network_df = pd.DataFrame(networks).drop_duplicates()
    print(f"Total interactions after deduplication: {network_df.shape[0]}")
    network_df.to_csv(output_path, index=False)
    print(f"Regulatory network saved to: {output_path}")
    return network_df

def process_motrpac_dataset(dataset_path, gene_map_path, token_dict_path, species=1):
    """
    Processes a MoTrPAC dataset by adding 'input_ids', 'values', 'length', and 'species' columns.
    """
    print("Loading dataset...")
    df = pd.read_csv(dataset_path, index_col=0)

    print("Loading gene map...")
    with open(gene_map_path, "rb") as f:
        gene_map = pickle.load(f)

    print("Loading token dictionary...")
    with open(token_dict_path, "rb") as f:
        token_dict = pickle.load(f)

    num_samples = df.shape[0]
    input_ids = np.zeros((num_samples, 2048), dtype=np.int32)
    values = np.zeros((num_samples, 2048), dtype=np.float32)
    lengths, species_list, gene_names_list = [], [], []

    print("Processing dataset...")
    unmatched_ensembl_ids = set()
    for array_idx, (_, row) in enumerate(tqdm(df.iterrows(), total=df.shape[0])):
        ensembl_ids = [strip_version(eid) for eid in row['feature_ids_human'].split(";")]
        expression_values = list(map(float, row['normalized_values'].split(";")))
        expression_dict = dict(zip(ensembl_ids, expression_values))

        tokens, current_values, current_gene_names = [], [], []
        for eid in ensembl_ids:
            if eid in token_dict:
                tokens.append(token_dict[eid])
                current_values.append(expression_dict.get(eid, 0.0))
                current_gene_names.append(gene_map.get(eid, "unknown"))
            else:
                tokens.append(0)
                current_values.append(0.0)
                current_gene_names.append("unknown")
                unmatched_ensembl_ids.add(eid)

        tokens = tokens[:2048]
        current_values = current_values[:2048]
        current_gene_names = current_gene_names[:2048]

        length = len(tokens)
        input_ids[array_idx, :length] = tokens
        values[array_idx, :length] = current_values
        lengths.append(length)
        species_list.append(species)
        gene_names_list.append(";".join(current_gene_names))

    output_df = pd.DataFrame({
        'input_ids': [row.tolist() for row in input_ids],
        'values': [row.tolist() for row in values],
        'length': lengths,
        'species': species_list,
        'gene_names': gene_names_list
    })
    return output_df

def convert_to_genecompass_format(processed_dataset, save_file_path):
    """
    Converts the processed dataset to GeneCompass-compatible Arrow format.
    """
    data_dict = {
        "input_ids": processed_dataset['input_ids'].tolist(),
        "values": processed_dataset['values'].tolist(),
        "length": [[length] for length in processed_dataset['length']],
        "species": [[species_id] for species_id in processed_dataset['species']],
    }
    
    features = Features({
        "input_ids": Sequence(feature=Value(dtype="int32")),
        "values": Sequence(feature=Value(dtype="float32")),
        "length": Sequence(feature=Value(dtype="int16")),
        "species": Sequence(feature=Value(dtype="int16")),
    })

    dataset = Dataset.from_dict(data_dict, features=features)
    dataset.save_to_disk(save_file_path)
       


def main():
    path = "/workspace/GeneCompass/"
    data_path = "/workspace/GeneCompass/data/MoTrPAC/MoTrPAC_data_Top2048_VAR_Global.csv"
    output_network_path = "/workspace/GeneCompass/data/MoTrPAC/unified_network.csv"
    tf_cache_path = "/workspace/GeneCompass/data/MoTrPAC/tf_list.csv"
    
    prior_knowledge_dir = f"{path}/downstream_tasks/grn_inference"
    gene_map_dict = f"{prior_knowledge_dir}/Gene_id_name_dict.pickle"
    gene_token_path = f"{prior_knowledge_dir}/h_m_token2000W.pickle"

    # Load or fetch transcription factors
    tf_list = load_or_fetch_tf_list(cache_path=tf_cache_path)
    print("Columns in TF list DataFrame:", tf_list.columns.tolist())

    if tf_list is None:
        print("Unable to proceed without transcription factor data.")
        return

    # Create the unified network
    create_unified_network(data_path, gene_map_dict, tf_list, output_network_path)
    print("Unified network construction complete.")

    dataset_path = f"{path}/data/"
    unified_matrix_path = f"{dataset_path}/unified_expression_matrix.csv"



    # Step 3: Create unified expression matrix from the raw dataset
    create_expression_matrix_from_raw(data_path, gene_map_dict, unified_matrix_path)
    print("Unified expression matrix generation complete.")


    output_dir = f"{dataset_path}/MoTrPAC/Subsets"
    group_columns = ["tissue", "sex", "intervention_type"]

    print("Splitting raw dataset...")
    split_by_groups(data_path, output_dir, group_columns)

    subset_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith("_raw.csv")]
    for subset_file in subset_files:
        
        subset_path = os.path.join(output_dir, subset_file)
        base_name = os.path.splitext(subset_file)[0]

        processed_dataset = process_motrpac_dataset(
            dataset_path=subset_file,
            gene_map_path=gene_map_dict,
            token_dict_path=gene_token_path,
            species=0
        )
        
        processed_output_dir = subset_file.replace("_raw.csv", "_arrow")
        os.makedirs(processed_output_dir, exist_ok=True)
        convert_to_genecompass_format(processed_dataset, processed_output_dir)

        

        # Generate Expression Matrix
        expression_matrix_path = os.path.join(output_dir, f"{base_name}_expression_matrix.csv")
        create_expression_matrix_from_subset(subset_path, gene_map_dict, expression_matrix_path)

        # Generate Regulatory Network
        network_path = os.path.join(output_dir, f"{base_name}_network.csv")
        create_network_for_subset(subset_path, gene_map_dict, tf_list, network_path)


        print(f"Converted subset saved to: {processed_output_dir}")

if __name__ == "__main__":
    main()
