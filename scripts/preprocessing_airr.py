#!/usr/bin/env python3
"""
Preprocess Data into TITAN input format.
Can be used in 2 modes.
    --input_type airr: Assumes that data is in standard AIRR format. Expects 
        separate files for each epitope with epitope name as filename.
    --input_type vdjdb: Assumes that data is in format as downloaded from 
        vdj database, with epitope sequences in "Epitope" column.
"""
import pandas as pd
import os
import csv
from pytoda.proteins import aas_to_smiles
import argparse
from paccmann_tcr.utils.utils import to_full_seq 

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_type', type=str,
    help='Either airr or vdjdb.'
)
parser.add_argument(
    '--infolder', type=str,
    help='Folder with all input files to preprocess.'
)
parser.add_argument(
    '--outfolder', type=str,
    help='Folder where preprocessed files are saved.'
)

ALLOWED_SYMBOLS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
V_J_SEGMENT_FOLDER = '../paccmann_tcr/utils/TCR_gene_segment_data'

def relabel(x):
    if x in [0,1]:
        return x
    else:
        if x == True:
            y = 1
        else:
            y = 0
        return y

# assign epitope ids
def assign_epi_ids(epitopes):
    epi_to_id = {}
    for i, epi in enumerate(epitopes):
        epi_to_id[epi] = i
    return epi_to_id

# assign TCR ids
def add_tcrs(tcr_to_id, max_tcr_id, raw_data, column_names):
    for index, row in raw_data.iterrows():
        identifier = (row[column_names[0]],row[column_names[1]],row[column_names[2]])
        if identifier not in tcr_to_id.keys():
            max_tcr_id += 1
            tcr_to_id[identifier] = max_tcr_id
    return tcr_to_id, max_tcr_id

def output_data(outfolder, epi_to_id, tcr_to_id, directory):
    # write out epitope and tcr files
    broken_seqs = []
    with open(os.path.join(outfolder, 'epitopes.csv'), 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for epitope in epi_to_id.keys():
            writer.writerow([epitope, epi_to_id[epitope]])

    with open(os.path.join(outfolder, 'epitopes.smi'), 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for epitope in epi_to_id.keys():
            writer.writerow([aas_to_smiles(epitope), epi_to_id[epitope]])

    with open(os.path.join(outfolder, 'tcr_full.csv'), 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for tcr_id in tcr_to_id.keys():
            (v, j, cdr3) = tcr_id
            fullseq = to_full_seq(directory, v, j, cdr3)
            for token in fullseq:
                if token not in ALLOWED_SYMBOLS:
                    print(fullseq, ' contains forbidden symbol ', token, ' Sequence will be skipped.')
                    broken_seqs.append(tcr_id)
            writer.writerow([fullseq, tcr_to_id[tcr_id]])

    with open(os.path.join(outfolder, 'tcr.csv'), 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for tcr_id in tcr_to_id.keys():
            (v, j, cdr3) = tcr_id
            writer.writerow([cdr3, tcr_to_id[tcr_id]])
    
    return broken_seqs

def main(
    input_type, infolder, outfolder
):
    tcr_to_id = {}
    max_tcr_id = 0

    if input_type == 'airr':
        column_names = ['v_call', 'j_call', 'junction_aa', 'Label']
        epitopes = [x.split('.')[0] for x in os.listdir(infolder)]
        
    if input_type == 'vdjdb':
        column_names = ['V', 'J', 'CDR3', 'Label']

    for i,filename in enumerate(os.listdir(infolder)):
        if not filename.startswith('.') and '.tsv' in filename or '.txt' in filename:
            print('Read in ',filename)
            raw_data = pd.read_table(os.path.join(infolder,filename))

            # Generate epitope and TCR IDs
            if input_type == 'airr':
                epitope = filename.split('.')[0]
                raw_data['Epitope'] = [epitope for x in raw_data[column_names[0]]]
            elif input_type == 'vdjdb':
                epitopes = list(set(raw_data['Epitope']))
            epi_to_id = assign_epi_ids(epitopes)
            tcr_to_id, max_tcr_id = add_tcrs(tcr_to_id, max_tcr_id, raw_data, column_names)

            # Generate full dataset
            interactions = pd.DataFrame()
            interactions['ligand_name'] = [epi_to_id[x] for x in raw_data['Epitope']]
            interactions['sequence_id'] = raw_data.apply(lambda x: tcr_to_id[(x[column_names[0]], x[column_names[1]], x[column_names[2]])], axis=1)
            interactions['label'] = [relabel(x) for x in raw_data[column_names[3]]]
            
            # Output Data
            broken_seqs = output_data(outfolder, epi_to_id, tcr_to_id, V_J_SEGMENT_FOLDER)
            # Remove interactions involving TCRs with broken sequence
            interactions = interactions[~interactions['sequence_id'].isin(broken_seqs)]
            interactions.to_csv(os.path.join(outfolder, filename.split('.')[0]+'.csv'))

if __name__ == '__main__':
    args = parser.parse_args()
    main(
        args.input_type, args.infolder,
        args.outfolder
    )


