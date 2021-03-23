[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# paccmann_tcr

TCell Receptor to peptide binding project

## Installation

The library itself has few dependencies (see [setup.py](setup.py)) with loose requirements. 

Create a conda environment:

```console
conda env create -f conda.yml
```

Activate the environment:

```console
conda activate titan
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
All data can be found in https://ibm.box.com/s/y7rpfxujoieurhtnjut4jw0q5g64mwtj .

## Example usage TITAN
The TITAN model uses the architecture published as 'paccmann_predictor' [package](https://github.com/PaccMann/paccmann_predictor). Example parameter files are given in the params folder.


```console
python3 scripts/flexible_training.py \
name_of_training_data_files.csv \
name_of_testing_data_files.csv \
path_to_tcr_file.csv \
path_to_epitope_file.csv/.smi \
path_to_pretrained_model \
path_to_store_finetuned_model \
training_name \
path_to_parameter_file \
bimodal_mca
```

## Example usage finetuning
To load a TITAN model after pretraining and finetune it on another dataset, use the `semifrozen_finetuning.py` script. Use the parameter `number_of_tunable_layers` to control the number of layers which will be tuned, the rest will be frozen. Model will freeze epitope input channel first and the final dense layers last. Do not change the input data type (i.e. SMILES or amino acids) between pretraining and finetuning.


```console
python3 scripts/semifrozen_finetuning.py \
name_of_training_data_files.csv \
name_of_testing_data_files.csv \
path_to_tcr_file.csv \
path_to_epitope_file.smi \
path_to_store_model \
path_to_parameter_file \
training_name \
bimodal_mca
```


## Example usage K-NN

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

## Citation
If you use `titan` in your projects, please cite the following:

*Citation will appear soon.*
