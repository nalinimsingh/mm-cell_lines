import os
import pandas as pd


def hgnc_to_ensembl(hgnc_df, map_dict):
    """Converts a DataFrame with HGNC_IDs to Ensembl_IDs.

    Removes unmapped columns.

    Args:
        hgnc_df (DataFrame): The DataFrame to be converted, with HGNC_IDs as column names
        map_dict (dict): Dictionary with HGNC_IDs as keys and Ensembl_IDs as values

    Returns:
        emsemble_df (DataFrame): The converted DataFrame, with Ensembl_IDs as column names
    """
    to_del = []
    for col in hgnc_df:
        if(col not in map_dict and col!='Unnamed: 0'):
            to_del.append(col)

    hgnc_df.drop(columns=to_del, inplace=True)

    ensembl_df = hgnc_df.rename(columns=map_dict)
    return ensembl_df


def get_map_dict(mapping_path):
    """Converts a CSV file mapping HGNC_IDs to Ensembl_IDs to a dictionary with the same mapping.

    Args:
        mapping_path (string): Path to CSV mapping file with columns 'HGNC_IDs' and 'Ensemble_IDs'
    Returns:
        gene_map_dict: Dictionary with HGNC_IDs as keys and Ensembl_IDs as values
    """
    gene_map = pd.read_csv(mapping_path)
    gene_map_dict = {}
    for _, row in gene_map.iterrows():
        gene_map_dict[row['HGNC_ID']] = row['Ensembl_ID']

    return gene_map_dict


def get_ccle_rnaseq_with_ensembl(rnaseq_path, mapping_path):
    """Wrapper function to read CCLE data with Ensembl_IDs.

    Args:
        rnaseq_path (string): Path to CCLE RNASeq data
        mapping_path (string): Path to CSV mapping file with columns 'HGNC_IDs' and 'Ensembl_IDs'
    Returns:
        cell_line_ensembl_rna (DataFrame): Output DataFrame with RNA Seq data, with Ensembl_IDs as column names
    """
    rnaseq = pd.read_csv(rnaseq_path)
    map_dict = get_map_dict(mapping_path)

    cell_line_ensembl_rna = hgnc_to_ensembl(rnaseq, map_dict)
    return cell_line_ensembl_rna
