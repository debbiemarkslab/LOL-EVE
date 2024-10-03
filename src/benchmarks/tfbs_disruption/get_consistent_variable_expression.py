import pandas as pd
import numpy as np
from scipy.stats import variation
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import math

def load_gtex_expression_data(file_path):
    """Load gene expression data from a gzipped GCT file."""
    with gzip.open(file_path, 'rt') as f:
        next(f)
        next(f)
        df = pd.read_csv(f, sep='\t', index_col=0)
    if 'Description' in df.columns:
        df = df.drop('Description', axis=1)
    return df

def load_biomart_data(file_path):
    """Load Biomart data for gene name conversion."""
    biomart_df = pd.read_csv(file_path, sep='\t')
    # Create a dictionary mapping Ensembl IDs to gene symbols
    gene_map = dict(zip(biomart_df['Gene stable ID'], biomart_df['Gene name']))
    return gene_map

def calculate_expression_variability(expression_data):
    """Calculate coefficient of variation for each gene across tissues."""
    cv = expression_data.apply(variation, axis=1)
    return cv.sort_values()

def get_consistent_and_variable_genes(cv, n_consistent=1000, n_variable=1000):
    """Get lists of genes with most consistent and most variable expression."""
    consistent_genes = cv.head(n_consistent).index.tolist()
    variable_genes = cv.tail(n_variable).index.tolist()
    return consistent_genes, variable_genes

def plot_expression_heatmap(expression_data, gene_list, title, gene_map=None):
    plt.figure(figsize=(12, len(gene_list) // 3))
    data_to_plot = expression_data.loc[gene_list]
    if gene_map:
        data_to_plot.index = [gene_map.get(gene, gene) for gene in data_to_plot.index]
    sns.heatmap(data_to_plot, cmap='YlOrRd', robust=True)
    plt.title(title)
    plt.xlabel('Tissues')
    plt.ylabel('Genes')
    plt.tight_layout()
    plt.show()

# Main analysis
gtex_file_path = "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"
biomart_file_path = "biomart_export.txt"  # Replace with your actual Biomart file path

expression_data = load_gtex_expression_data(gtex_file_path)
gene_map = load_biomart_data(biomart_file_path)

cv = calculate_expression_variability(expression_data)

cv.dropna(inplace=True)

consistent_genes, variable_genes = get_consistent_and_variable_genes(cv)

# Convert Ensembl IDs to gene symbols
consistent_genes_symbols = [gene_map.get(gene, gene) for gene in consistent_genes]
variable_genes_symbols = [gene_map.get(gene, gene) for gene in variable_genes]

consistent_genes = [g.split('.')[0] for g in consistent_genes]
variable_genes = [g.split('.')[0] for g in variable_genes]

# Convert Ensembl IDs to gene symbols
consistent_genes_symbols = [gene_map.get(gene, gene) for gene in consistent_genes]
variable_genes_symbols = [gene_map.get(gene, gene) for gene in variable_genes]


consistent_genes_symbols = [x for x in consistent_genes_symbols if isinstance(x, str) or not math.isnan(x)]
variable_genes_symbols = [x for x in variable_genes_symbols if isinstance(x, str) or not math.isnan(x)]

print(len(consistent_genes_symbols))
print(len(variable_genes_symbols))

# Save results to files
pd.DataFrame({'Gene Symbol': consistent_genes_symbols[0:500]}).to_csv("gtex_consistent_genes_500.txt", index=False)
pd.DataFrame({'Gene Symbol': variable_genes_symbols[0:500]}).to_csv("gtex_variable_genes_500.txt", index=False)
