[![Python package](https://github.com/PaccMann/TITAN/actions/workflows/python-package.yml/badge.svg)](https://github.com/PaccMann/TITAN/actions/workflows/python-package.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# TITAN
 TITAN - **T**cr ep**IT**ope bimodal **A**ttention **N**etworks

## Installation

The library itself has few dependencies (see [setup.py](setup.py)) with loose requirements. 

Create a virtual environment and install dependencies

```console
python -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```
Install in editable mode for development:
```console
pip install -e .
```


## Data structure
For data handling, we make use of the `pytoda` [package](https://github.com/PaccMann/paccmann_datasets).
If you bring your own data, it needs to adhere to the following format:
- `tcrs.csv`        A `.csv` file containing two columns, one for the tcr sequences and one for their IDs.
- `epitopes.csv`    A `.csv` file containing two columns, one for the epitope sequences and one for their IDs.
    This can optionally also be a `.smi` file (tab-separated) with the SMILES seuqences of the eptiopes.
- `train.csv`       A `.csv` file containing three columns, one for TCR IDs, one for epitope IDs and one for the labels. This data is used for training.
- `test.csv`       A `.csv` file containing three columns, one for TCR IDs, one for epitope IDs and one for the labels. This data is used for testing.

NOTE: `tcrs.csv` and `epitopes.csv` need to contain **all** TCRs and epitopes used during training and testing. No duplicates in both sequence and IDs are allowed.
All data can be found in https://ibm.box.com/v/titan-dataset .


# Example usages
## Train a TITAN model
The TITAN model uses the architecture published as 'paccmann_predictor' [package](https://github.com/PaccMann/paccmann_predictor). Example parameter files are given in the params folder.


```console
python3 scripts/flexible_training.py \
name_of_training_data_files.csv \
name_of_testing_data_files.csv \
path_to_tcr_file.csv \
path_to_epitope_file.csv/.smi \
path_to_store_trained_model \
path_to_parameter_file \
training_name \
bimodal_mca
```

## Finetune an existing TITAN model
To load a TITAN model after pretraining and finetune it on another dataset, use the `semifrozen_finetuning.py` script. Use the parameter `number_of_tunable_layers` to control the number of layers which will be tuned, the rest will be frozen. Model will freeze epitope input channel first and the final dense layers last. Do not change the input data type (i.e. SMILES or amino acids) between pretraining and finetuning.


```console
python3 scripts/semifrozen_finetuning.py \
name_of_training_data_files.csv \
name_of_testing_data_files.csv \
path_to_tcr_file.csv \
path_to_epitope_file.smi \
path_to_pretrained_model \
path_to_store_model \
training_name \
path_to_parameter_file \
bimodal_mca
```

## Run trained TITAN model on data
A trained model is provided in trained_model. The model is pretrained on BindingDB and finetuned using the semifrozen setting, on full TCR sequences and with SMILES encoding of epitopes. All parameters can be found in the parameter files provided. 

```console
python3 scripts/flexible_model_eval.py \
name_of_test_data_file.csv \
path_to_tcr_file.csv \
path_to_epitope_file.smi \
path_to_trained_model_folder \
bimodal_mca \
save_name
```

## Evaluate K-NN baseline on cross validation

The script `scripts/knn_cv.py` uses the KNN baseline model of the paper and performs a cross validation.
The script can be used in two modes, *shared* and *separate*. *Shared* is the default mode as specified [above](#Data-structure). In *separate* mode, the TCRs and epitope sequences for training and testing dont need to be in the same file, but can be split across two files. To use this mode, simply provide additional paths to `-test_tcr` and `test_ep` arguments.


```console
python3 scripts/knn_cv.py \
-d path_to_data_folder \
-tr name_of_training_data_files.csv \
-te name_of_testing_data_files.csv \
-f 10 \
-ep path_to_epitope_file.ccsv \
-tcr path_to_tcr_file.csv \
-r path_to_result_folder \
-k 25
```
type `python3 scripts/knn_cv.py -h` for help.
The data in `data_folder` needs to be structured as:

```console
data_path
├── fold0
│   ├── name_of_training_data_files.csv
│   ├── name_of_testing_data_files.csv
...
├── fold9
│   ├── name_of_training_data_files.csv
│   ├── name_of_testing_data_files.csv
```

## Data Handling
To generate full sequences of TCRs from CDR3 sequence and V and J segment names, the `cdr3_to_full_seq.py` script can be used. The script relies on the user having downloaded a fasta files containing the Names of V and J segments with their respecive sequences called `V_segment_sequences.fasta` and `J_segment_sequences.fasta`. These can be downloaded from IMGT.org. Header names must be provided to the script to adapt to different format of the input file.

```console
python3 scripts/cdr3_to_full_seq.py \
directoy_with_VJ_segment_fasta_files \
path_to_file_with_input_sequences.csv \
v_seq_header \
j_seq_header \
cdr3_header \
path_to_output_file.csv
```


## Citation
If you use `titan` in your projects, please cite the following:

```bib
@article{weber2021titan
    author = {Weber, Anna and Born, Jannis and Rodriguez Martinez, Maria},
    title = "{TITAN: T-cell receptor specificity prediction with bimodal attention networks}",
    journal = {Bioinformatics},
    volume = {37},
    number = {Supplement_1},
    pages = {i237-i244},
    year = {2021},
    month = {07},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btab294},
    url = {https://doi.org/10.1093/bioinformatics/btab294}
}
```
