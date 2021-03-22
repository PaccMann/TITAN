#!/usr/bin/env python3
"""Finetune affinity predictor by optionally freezing the ligand/epitope branch"""
import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from time import time

import numpy as np
import pandas as pd
import torch
from paccmann_predictor.models import MODEL_FACTORY
from paccmann_predictor.utils.hyperparams import OPTIMIZER_FACTORY
from paccmann_predictor.utils.utils import get_device
from pytoda.datasets import DrugAffinityDataset
from pytoda.proteins import ProteinFeatureLanguage, ProteinLanguage
from pytoda.smiles.smiles_language import SMILESTokenizer
from sklearn.metrics import (
    auc, average_precision_score, precision_recall_curve, roc_curve
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

from paccmann_tcr.utils.transferlearning import update_mca_model

torch.manual_seed(123456)

# setup logging
logging.basicConfig(stream=sys.stdout)

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument(
    'train_affinity_filepath', type=str,
    help='Path to the affinity data.'
)
parser.add_argument(
    'test_affinity_filepath', type=str,
    help='Path to the affinity data.'
)
parser.add_argument(
    'receptor_filepath', type=str,
    help='Path to the receptor aa data (.csv)'
)
parser.add_argument(
    'ligand_filepath', type=str,
    help='Path to the ligand data (.smi)'
)
parser.add_argument(
    'pretrained_model_path', type=str,
    help='Directory to read the model from'
)
parser.add_argument(
    'finetuned_model_path', type=str,
    help='Directory to store the finetuned model in'
)
parser.add_argument(
    'training_name', type=str,
    help='Name for the training.'
)
parser.add_argument(
    'params_filepath', type=str,
    help='Path to the parameter file for finetuning'
)
parser.add_argument(
    'model_type', type=str, default='bimodel_mca',
    help='Name model type you want to use: bimodal_mca, context_encoding_mca.'
)
# yapf: enable


def main(
    train_affinity_filepath, test_affinity_filepath, receptor_filepath,
    ligand_filepath, pretrained_model_path, finetuned_model_path,
    training_name, model_type, params_filepath
):

    logger = logging.getLogger(f'{training_name}')
    logger.setLevel(logging.DEBUG)

    # Create model directory and dump files
    model_dir = os.path.join(finetuned_model_path, training_name)
    os.makedirs(os.path.join(model_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'results'), exist_ok=True)

    # Process and dump parameter file:
    params = {}
    with open(os.path.join(pretrained_model_path, 'model_params.json')) as fp:
        params.update(json.load(fp))
    with open(os.path.join(model_dir, 'model_params.json'), 'w') as fp:
        json.dump(params, fp, indent=4)

    finetune_params = {}
    with open(params_filepath) as fp:
        finetune_params.update(json.load(fp))
    with open(os.path.join(model_dir, 'finetune_params.json'), 'w') as fp:
        json.dump(finetune_params, fp, indent=4)

    device = get_device()

    # Load languages
    smiles_language = SMILESTokenizer.from_pretrained(
        os.path.join(pretrained_model_path)
    )
    smiles_language.set_encoding_transforms(
        randomize=None,
        add_start_and_stop=params.get('ligand_start_stop_token', True),
        padding=params.get('ligand_padding', True),
        padding_length=params.get('ligand_padding_length', 500),
        device=device,
    )
    # Set transform
    test_smiles_language = deepcopy(smiles_language)

    smiles_language.set_smiles_transforms(
        augment=finetune_params.get('augment_smiles', False),
        canonical=finetune_params.get('smiles_canonical', False),
        kekulize=finetune_params.get('smiles_kekulize', False),
        all_bonds_explicit=finetune_params.get('smiles_bonds_explicit', False),
        all_hs_explicit=finetune_params.get('smiles_all_hs_explicit', False),
        remove_bonddir=finetune_params.get('smiles_remove_bonddir', False),
        remove_chirality=finetune_params.get('smiles_remove_chirality', False),
        selfies=finetune_params.get('selfies', False),
        sanitize=finetune_params.get('selfies', False)
    )
    test_smiles_language.set_smiles_transforms(
        augment=False,
        canonical=finetune_params.get('test_smiles_canonical', False),
        kekulize=finetune_params.get('smiles_kekulize', False),
        all_bonds_explicit=finetune_params.get('smiles_bonds_explicit', False),
        all_hs_explicit=finetune_params.get('smiles_all_hs_explicit', False),
        remove_bonddir=finetune_params.get('smiles_remove_bonddir', False),
        remove_chirality=finetune_params.get('smiles_remove_chirality', False),
        selfies=finetune_params.get('selfies', False),
        sanitize=finetune_params.get('selfies', False)
    )

    if params.get('receptor_embedding', 'learned') == 'predefined':
        protein_language = ProteinFeatureLanguage.load(
            os.path.join(pretrained_model_path, 'protein_language.pkl')
        )
    else:
        protein_language = ProteinLanguage.load(
            os.path.join(pretrained_model_path, 'protein_language.pkl')
        )
    smiles_language.save_pretrained(os.path.join(model_dir))
    protein_language.save(os.path.join(model_dir, 'protein_language.pkl'))

    # Restore model
    logger.info("Restore model...")
    model_fn = params.get('model_fn', model_type)
    weights_path = os.path.join(
        pretrained_model_path, 'weights', f'best_ROC-AUC_{model_fn}.pt'
    )
    model = MODEL_FACTORY[model_fn](params).to(device)
    try:
        model.load(weights_path, map_location=device)
    except Exception:
        raise TypeError('Error in model restoring.')
    model = update_mca_model(model, finetune_params)

    num_params = sum(p.numel() for p in model.parameters())
    num_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Number of parameters: {num_params}, trainable: {num_train}')

    logger.info("Model restored, set up dataset...")
    # Assemble datasets
    train_dataset = DrugAffinityDataset(
        drug_affinity_filepath=train_affinity_filepath,
        smi_filepath=ligand_filepath,
        protein_filepath=receptor_filepath,
        smiles_language=smiles_language,
        protein_language=protein_language,
        smiles_padding=params.get('ligand_padding', True),
        smiles_padding_length=params.get('ligand_padding_length', None),
        smiles_add_start_and_stop=params.get('ligand_add_start_stop', True),
        protein_amino_acid_dict=params.get('protein_amino_acid_dict', 'iupac'),
        protein_padding=params.get('receptor_padding', True),
        protein_padding_length=params.get('receptor_padding_length', None),
        protein_add_start_and_stop=params.get('receptor_add_start_stop', True),
        protein_augment_by_revert=finetune_params.get(
            'protein_augment', False
        ),
        device=device,
        drug_affinity_dtype=torch.float,
        backend='eager'
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=finetune_params['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=finetune_params.get('num_workers', 0)
    )

    test_dataset = DrugAffinityDataset(
        drug_affinity_filepath=test_affinity_filepath,
        smi_filepath=ligand_filepath,
        protein_filepath=receptor_filepath,
        smiles_language=test_smiles_language,
        protein_language=protein_language,
        smiles_padding=params.get('ligand_padding', True),
        smiles_padding_length=params.get('ligand_padding_length', None),
        smiles_add_start_and_stop=params.get('ligand_add_start_stop', True),
        protein_amino_acid_dict=params.get('protein_amino_acid_dict', 'iupac'),
        protein_padding=params.get('receptor_padding', True),
        protein_padding_length=params.get('receptor_padding_length', None),
        protein_add_start_and_stop=params.get('receptor_add_start_stop', True),
        protein_augment_by_revert=False,
        device=device,
        drug_affinity_dtype=torch.float,
        backend='eager'
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        drop_last=True,
        num_workers=params.get('num_workers', 0)
    )
    logger.info(
        f'Training dataset has {len(train_dataset)} samples, test set has '
        f'{len(test_dataset)}.'
    )

    logger.info(
        f'Loader length: Train - {len(train_loader)}, test - {len(test_loader)}'
    )
    save_top_model = os.path.join(model_dir, 'weights/{}_{}_{}.pt')

    min_loss, max_roc_auc = 100, 0

    # Define optimizer
    optimizer = (
        OPTIMIZER_FACTORY[finetune_params.get('optimizer', 'adam')](
            model.parameters(),
            lr=finetune_params.get('lr', 0.0001),
            weight_decay=finetune_params.get('weight_decay', 0.001)
        )
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.8,
        patience=3,
        min_lr=1e-07,
        verbose=True
    )

    def save(path, metric, typ, val=None):
        """Routine to save model"""
        model.save(path.format(typ, metric, model_fn))
        info = {'best_roc_auc': str(max_roc_auc), 'test_loss': str(min_loss)}
        with open(
            os.path.join(model_dir, 'results', metric + '.json'), 'w'
        ) as f:
            json.dump(info, f)

        pd.DataFrame({
            'labels': labels,
            'predictions': predictions
        }).to_csv(os.path.join(model_dir, 'results', metric + '_preds.csv'))

        if typ == 'best':
            logger.info(
                f'\t New best performance in "{metric}"'
                f' with value : {val:.7f} in epoch: {epoch}'
            )

    # Start training
    logger.info('Training about to start...\n')
    t = time()
    loss_training = []
    loss_validation = []
    # model.save(save_top_model.format('epoch', '0', model_fn))

    num_epochs = finetune_params.get('epochs', 200)
    logger.info(
        train_dataset.smiles_dataset.smiles_language.transform_encoding
    )
    logger.info(train_dataset.smiles_dataset.smiles_language.transform_smiles)
    logger.info(test_dataset.smiles_dataset.smiles_language.transform_encoding)
    logger.info(test_dataset.smiles_dataset.smiles_language.transform_smiles)
    logger.info(train_dataset.protein_sequence_dataset.language_transforms)
    logger.info(test_dataset.protein_sequence_dataset.language_transforms)

    for epoch in range(num_epochs):

        logger.info(f"== Epoch [{epoch}/{num_epochs}] ==")
        t = time()

        # Measure validation performance
        model.eval()
        with torch.no_grad():
            test_loss = 0
            predictions = []
            labels = []
            for ind, (ligand, receptors, y) in enumerate(test_loader):
                torch.cuda.empty_cache()
                y_hat, pred_dict = model(
                    ligand.to(device), receptors.to(device)
                )
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
            f"\t **** TESTING **** Epoch [{epoch + 1}/{num_epochs}], "
            f"loss: {test_loss:.5f}, ROC-AUC: {test_roc_auc:.3f}, "
            f"Average precision: {avg_precision:.3f}."
        )

        if test_roc_auc > max_roc_auc:
            max_roc_auc = test_roc_auc
            save(save_top_model, 'ROC-AUC', 'best', max_roc_auc)
            ep_roc = epoch
            roc_auc_loss = test_loss
            roc_auc_pr = avg_precision

        if test_loss < min_loss:
            min_loss = test_loss
            save(save_top_model, 'loss', 'best', min_loss)
            ep_loss = epoch
            loss_roc_auc = test_roc_auc
        if (epoch + 1) % finetune_params.get('save_model', 1000) == 0:
            save(save_top_model, 'epoch', str(epoch))

        # Optionally reduce LR
        if epoch > 0:
            scheduler.step(test_roc_auc)

        # Now training
        model.train()
        train_loss = 0

        for ind, (ligand, receptors, y) in enumerate(train_loader):

            torch.cuda.empty_cache()
            if ind % 100 == 0:
                logger.info(f'Batch {ind}/{len(train_loader)}')
            y_hat, pred_dict = model(ligand, receptors)
            loss = model.loss(y_hat, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            # Apply gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(),1e-6)
            optimizer.step()
            train_loss += loss.item()

        logger.info(
            "\t **** TRAINING ****   "
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"loss: {train_loss / len(train_loader):.5f}. "
            f"This took {time() - t:.1f} secs."
        )
        loss_training.append(train_loss / len(train_loader))

    logger.info(
        'Overall best performances are: \n \t'
        f'Loss = {min_loss:.4f} in epoch {ep_loss} '
        f'\t (ROC-AUC was {loss_roc_auc:4f}) \n \t'
        f'ROC-AUC = {max_roc_auc:.4f} in epoch {ep_roc} '
        f'\t (Loss was {roc_auc_loss:4f})'
    )
    save(save_top_model, 'training', 'done')
    logger.info('Done with training, models saved, shutting down.')

    np.save(
        os.path.join(model_dir, 'results', 'loss_training.npy'), loss_training
    )
    np.save(
        os.path.join(model_dir, 'results', 'loss_validation.npy'),
        loss_validation
    )

    # save best results
    result_file = os.path.join(model_dir, 'results', 'overview.csv')
    with open(result_file, "a") as myfile:
        myfile.write(
            f'{training_name},{max_roc_auc:.4f},{roc_auc_pr:.4f},'
            f'{roc_auc_loss:.4f},{ep_roc} \n \t'
        )


if __name__ == '__main__':
    args = parser.parse_args()
    main(
        args.train_affinity_filepath, args.test_affinity_filepath,
        args.receptor_filepath, args.ligand_filepath,
        args.pretrained_model_path, args.finetuned_model_path,
        args.training_name, args.model_type, args.params_filepath
    )
