# mm-cell_lines

Deep representation learning for predicting progression-free survival in multiple myeloma patients.

keywords: multiple myeloma, representation learning, RNA-seq

## Dependencies/Startup
In order to run this code, you will need to install the [ml-mmrf library](https://github.com/clinicalml/ml_mmrf). 

You will also need to download the following files from the MMRF researcher gateway:
- `CoMMpass_IA15_FlatFiles.tar.gz` (unzipped)
- `MMRF_CoMMpass_IA15a_E74GTF_Salmon_Gene_TPM.txt`
- `MMRF_OS_PFS_ASCT.csv`
- `MMRF_OS_PFS_non-ASCT.csv`

Build the MMRF data by navigating to ml_mmrf/core and running the following two commands:

`python build_mmrf_dataset.py ––fdir [YOUR FOLDER] --outcomes_type pfs --ia_version IA15 --recreate_splits True`

`python build_mmrf_dataset.py ––fdir [YOUR FOLDER] --outcomes_type pfs --ia_version IA15 --recreate_splits False`


To use the CCLE data, you will need the following files from the [DepMap dataset](https://depmap.org/portal/download/):
- `sample_info.csv`
- `CCLE_expression.csv`
- `Achilles_gene_effect.csv`*
- `sanger-dose-response.csv`

*Due to a recent renaming, you may need to download `CRISPR_gene_effect.csv`, instead.


## Reproducing Experiments
The core experiments for this project are found in 3 separate Jupyter notebooks, each of which can be run end-to-end. More detailed comments are available within each Jupyter notebook.
- `Autoencoders.ipynb`: Trains and evaluates various autoencoders and downstream classifiers, and writes the embeddings to the `.h5` files in `autoencoder_embeddings/` (used by downstream notebooks).
- `CCLE_transfer.ipynb`: Trains and evaluates classifiers for IC50 prediction-based transfer experiments.
- `Cell_perturbation.ipynb`: Loads and evaluates classifiers for gene perturbation-based transfer experiments.

The `requirements.txt` file for conda environments used to run each of these sets of experiments are found in `requirements/`.

Trained versions of our best models are also available in `trained_models`.
