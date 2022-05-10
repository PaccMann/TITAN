python3 ../scripts/flexible_training.py \
data/fold0/train_small.csv \
data/fold0/test_small.csv  \
data/tcr.csv \
data/epitopes.csv \
trained_model \
data/params_small.json \
tutorial_setting \
bimodal_mca

#python3 ../scripts/semifrozen_finetuning.py \
#data/fold0/train.csv \
#data/fold0/test.csv  \
#data/tcr.csv \
#data/epitopes.smi \
#'/Users/wbr/Library/CloudStorage/Box-Box/Molecular_SysBio/data/paccmann/TITAN/public/trained_model' \
#trained_model \
#finetuned_model \
#../params/params_finetuning.json \
#bimodal_mca