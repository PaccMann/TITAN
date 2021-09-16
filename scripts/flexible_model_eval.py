#!/usr/bin/env python3
"""Train Affinity predictor model."""
import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from paccmann_predictor.models import MODEL_FACTORY
from paccmann_predictor.utils.utils import get_device
from pytoda.datasets import (
    DrugAffinityDataset, ProteinProteinInteractionDataset
)
from pytoda.proteins import ProteinFeatureLanguage, ProteinLanguage
from pytoda.smiles.smiles_language import SMILESTokenizer
from sklearn.metrics import (
    auc, average_precision_score, precision_recall_curve, roc_curve
)

torch.manual_seed(123456)

# setup logging
logging.basicConfig(stream=sys.stdout)

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument(
    'test_affinity_filepath', type=str,
    help='Path to the affinity data.'
)
parser.add_argument(
    'receptor_filepath', type=str,
    help='Path to the receptor aa data. (.csv)'
)
parser.add_argument(
    'ligand_filepath', type=str,
    help='Path to the ligand data. (SMILES .smi or aa .csv)'
)
parser.add_argument(
    'model_path', type=str,
    help='Directory from where the model will be loaded.'
)
parser.add_argument(
    'model_type', type=str,
    help='Name model type you want to use: bimodal_mca, context_encoding_mca.'
)
parser.add_argument(
    'save_name', type=str,
    help='Name you want to save results under.'
)
# yapf: enable


def main(
    test_affinity_filepath, receptor_filepath, ligand_filepath, model_path,
    model_type, save_name
):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Process parameter file:
    params_filepath = os.path.join(model_path, 'model_params.json')
    params = {}
    with open(params_filepath) as fp:
        params.update(json.load(fp))

    device = get_device()

    # Load languages

    smiles_language = SMILESTokenizer.from_pretrained(model_path)
    smiles_language.set_encoding_transforms(
        randomize=None,
        add_start_and_stop=params.get('ligand_start_stop_token', True),
        padding=params.get('ligand_padding', True),
        padding_length=params.get('ligand_padding_length', True),
        device=device,
    )
    smiles_language.set_smiles_transforms(
        augment=False,
        canonical=params.get('smiles_canonical', False),
        kekulize=params.get('smiles_kekulize', False),
        all_bonds_explicit=params.get('smiles_bonds_explicit', False),
        all_hs_explicit=params.get('smiles_all_hs_explicit', False),
        remove_bonddir=params.get('smiles_remove_bonddir', False),
        remove_chirality=params.get('smiles_remove_chirality', False),
        selfies=params.get('selfies', False),
        sanitize=params.get('sanitize', False)
    )
    if params.get('receptor_embedding', 'learned') == 'predefined':
        protein_language = ProteinFeatureLanguage.load(
            os.path.join(model_path, 'protein_language.pkl')
        )
    else:
        protein_language = ProteinLanguage.load(
            os.path.join(model_path, 'protein_language.pkl')
        )

    # Prepare the dataset
    logger.info("Start data preprocessing...")

    # Check if ligand as SMILES or as aa
    ligand_name, ligand_extension = os.path.splitext(ligand_filepath)
    if ligand_extension == '.csv':
        logger.info(
            'ligand file has extension .csv \n'
            'Please make sure ligand is provided as amino acid sequence.'
        )
        test_dataset = ProteinProteinInteractionDataset(
            sequence_filepaths=[[ligand_filepath], [receptor_filepath]],
            entity_names=['ligand_name', 'sequence_id'],
            labels_filepath=test_affinity_filepath,
            annotations_column_names=['label'],
            protein_language=protein_language,
            amino_acid_dict='iupac',
            padding_lengths=[
                params.get('ligand_padding_length', None),
                params.get('receptor_padding_length', None)
            ],
            paddings=params.get('ligand_padding', True),
            add_start_and_stops=params.get('add_start_stop_token', True),
            augment_by_reverts=params.get('augment_test_data', False),
            randomizes=False,
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            drop_last=True,
            num_workers=params.get('num_workers', 0)
        )

    elif ligand_extension == '.smi':
        logger.info(
            'ligand file has extension .smi \n'
            'Please make sure ligand is provided as SMILES.'
        )

        test_dataset = DrugAffinityDataset(
            drug_affinity_filepath=test_affinity_filepath,
            smi_filepath=ligand_filepath,
            protein_filepath=receptor_filepath,
            smiles_language=smiles_language,
            protein_language=protein_language,
            smiles_padding=params.get('ligand_padding', True),
            smiles_padding_length=params.get('ligand_padding_length', None),
            smiles_add_start_and_stop=params.get(
                'ligand_add_start_stop', True
            ),
            smiles_augment=False,
            smiles_canonical=params.get('test_smiles_canonical', False),
            smiles_kekulize=params.get('smiles_kekulize', False),
            smiles_all_bonds_explicit=params.get(
                'smiles_bonds_explicit', False
            ),
            smiles_all_hs_explicit=params.get('smiles_all_hs_explicit', False),
            smiles_remove_bonddir=params.get('smiles_remove_bonddir', False),
            smiles_remove_chirality=params.get(
                'smiles_remove_chirality', False
            ),
            smiles_selfies=params.get('selfies', False),
            protein_amino_acid_dict=params.get(
                'protein_amino_acid_dict', 'iupac'
            ),
            protein_padding=params.get('receptor_padding', True),
            protein_padding_length=params.get('receptor_padding_length', None),
            protein_add_start_and_stop=params.get(
                'receptor_add_start_stop', True
            ),
            protein_augment_by_revert=False,
            device=device,
            drug_affinity_dtype=torch.float,
            backend='eager',
            iterate_dataset=False
        )
        logger.info(f'Test dataset has {len(test_dataset)} samples.')
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            drop_last=True,
            num_workers=params.get('num_workers', 0)
        )
        logger.info(
            f'ligand_vocabulary_size  {smiles_language.number_of_tokens} '
            f'receptor_vocabulary_size {protein_language.number_of_tokens}.'
        )

    else:
        raise ValueError(
            f"Choose ligand_filepath with extension .csv or .smi, \
        given was {ligand_extension}"
        )
    logger.info(f'Test dataset has {len(test_dataset)} samples.')

    model_fn = params.get('model_fn', model_type)
    model = MODEL_FACTORY[model_fn](params).to(device)
    model._associate_language(smiles_language)
    model._associate_language(protein_language)

    model_file = os.path.join(
        model_path, 'weights', 'done_training_bimodal_mca.pt'
    )

    logger.info(f'looking for model in {model_file}')

    if os.path.isfile(model_file):
        logger.info('Found existing model, restoring now...')
        model.load(model_file, map_location=device)

        logger.info(f'model loaded: {model_file}')

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Number of parameters: {num_params}')

    # Measure validation performance
    loss_validation = []
    model.eval()
    with torch.no_grad():
        test_loss = 0
        predictions = []
        labels = []
        for ind, (ligand, receptors, y) in enumerate(test_loader):
            torch.cuda.empty_cache()
            y_hat, pred_dict = model(ligand.to(device), receptors.to(device))
            predictions.append(y_hat)
            labels.append(y.clone())
            loss = model.loss(y_hat, y.to(device))
            test_loss += loss.item()

    predictions = torch.cat(predictions, dim=0).flatten().cpu().numpy()
    labels = torch.cat(labels, dim=0).flatten().cpu().numpy()
    loss_validation.append(test_loss / len(test_loader))

    test_loss = test_loss / len(test_loader)
    fpr, tpr, _ = roc_curve(labels, predictions)
    test_roc_auc = auc(fpr, tpr)

    # calculations for visualization plot
    precision, recall, _ = precision_recall_curve(labels, predictions)
    avg_precision = average_precision_score(labels, predictions)

    logger.info(
        f"\t **** TESTING **** loss: {test_loss:.5f}, "
        f"ROC-AUC: {test_roc_auc:.3f}, Average precision: {avg_precision:.3f}."
    )

    np.save(
        os.path.join(model_path, 'results', save_name + '.npy'),
        np.vstack([predictions, labels])
    )


if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    # run the training
    main(
        args.test_affinity_filepath, args.receptor_filepath,
        args.ligand_filepath, args.model_path, args.model_type, args.save_name
    )
