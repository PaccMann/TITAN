#!/usr/bin/env python3
"""
Evaluate lazy KNN baseline predictor in a Cross-Validation setting.
Can be used in 2 modes.
    - shared    : Assumes that epitope_fp and tcr_fp contain all epitopes and tcrs that
        are used in training and testng dataset.
    - separate  : Assumes that two paths are passed for both TCRs and epitopes.
        - epitope_fp: Path to file containing all epitopes used in training.
        - epitope_test_fp: Path to file containing all epitopes used in testing.
        - tcr_fp: Path to file containing all TCRs used in training.
        - tcr_test_fp: Path to file containing all TCRs used in testing.

The default mode is 'shared' which is used iff the two optional args '-test_ep' and
'-test_tcr' are left empty (i.e., their default).
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from pytoda.files import read_smi
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, balanced_accuracy_score,
    confusion_matrix, matthews_corrcoef, precision_score, roc_curve
)

from paccmann_tcr.models import knn
from paccmann_tcr.utils.plot_knn import plot_roc_prc
from paccmann_tcr.utils.utils import cutoff_youdens_j

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument(
    '-d',
    '--data_path',
    type=str,
    help='Path to the folder where the fold-specific data is stored',
)
parser.add_argument(
    '-tr',
    '--train_name',
    type=str,
    help='Name of train files stored _inside_ the fold-specific folders',
)
parser.add_argument(
    '-te',
    '--test_name',
    type=str,
    help='Name of test files stored _inside_ the fold-specific folders',
)
parser.add_argument(
    '-f',
    '--num_folds',
    type=int,
    help='Number of folds. Folders should be named fold0, fold1 etc.  ',
)
parser.add_argument(
    '-ep', '--epitope_fp', type=str, help='Path to the epitope data (.csv)'
)
parser.add_argument(
    '-tcr', '--tcr_fp', type=str, help='Path to the epitope data (.csv)'
)
parser.add_argument(
    '-r', '--results_path', type=str, help='Path to folder to store results'
)
parser.add_argument(
    '-k',
    '--k',
    type=int,
    help=(
        'K for the KNN classification. Note that classification reports are generated'
        ' for all odd x: 1<= x <= k'
    ),
    default=15,
)
parser.add_argument(
    '-test_ep',
    '--test_epitope_fp',
    type=str,
    default='.',
    required=False,
    help='Path to the test epitope data (.csv)'
)
parser.add_argument(
    '-test_tcr',
    '--test_tcr_fp',
    type=str,
    required=False,
    default='.',
    help='Path to the test tcr data (.csv)'
)


def main(
    data_path, train_name, test_name, num_folds, epitope_fp, tcr_fp,
    results_path, k, test_epitope_fp, test_tcr_fp
):

    logger = logging.getLogger('knn_prediction')
    logging.getLogger('matplotlib.font_manager').disabled = True

    train_epi = read_smi(epitope_fp, names=['data'])
    train_tcr = read_smi(tcr_fp, names=['data'])

    test_epi = train_epi if test_epitope_fp == '.' else read_smi(
        test_epitope_fp, names=['data']
    )
    test_tcr = train_tcr if test_tcr_fp == '.' else read_smi(
        test_tcr_fp, names=['data']
    )

    # Create right amount of empty lists
    all_results = np.empty((np.ceil(k / 2).astype(int), 0)).tolist()

    for fold in range(num_folds):
        train_data = pd.read_csv(
            os.path.join(data_path, f'fold{fold}', train_name), index_col=0
        )
        test_data = pd.read_csv(
            os.path.join(data_path, f'fold{fold}', test_name), index_col=0
        )
        train_epitopes = train_epi.loc[train_data['ligand_name']]['data']
        train_tcrs = train_tcr.loc[train_data['sequence_id']]['data']
        test_epitopes = test_epi.loc[test_data['ligand_name']]['data']
        test_tcrs = test_tcr.loc[test_data['sequence_id']]['data']
        train_labels = train_data['label']
        test_labels = test_data['label']
        logger.info(f'Fold {fold}: Data loaded')
        # Classify data
        predictions, knn_labels = knn(
            train_epitopes,
            train_tcrs,
            train_labels,
            test_epitopes,
            test_tcrs,
            k=k,
            return_knn_labels=True,
        )
        logger.info(f'Fold {fold}: Predictions done')

        for idx, _k in enumerate(range(k, 0, -2)):

            # Overwrite predictions to match _k instead of k
            predictions = [
                np.mean(sample_knns[:_k]) for sample_knns in knn_labels
            ]

            # Compute metrics
            #   Continuous metrics
            fpr, tpr, thresholds = roc_curve(test_labels, predictions)
            roc_auc = auc(fpr, tpr)
            precision_recall_score = average_precision_score(
                test_labels, predictions
            )

            #   Metrics for a truly binary classifier
            theta = cutoff_youdens_j(fpr, tpr, thresholds)
            binary_predictions = [p > theta for p in predictions]
            accuracy = accuracy_score(test_labels, binary_predictions)
            balanced_accuracy = balanced_accuracy_score(
                test_labels, binary_predictions
            )
            precision = precision_score(test_labels, binary_predictions)
            mcc = matthews_corrcoef(test_labels, binary_predictions)
            tn, fp, fn, tp = confusion_matrix(test_labels,
                                              binary_predictions).ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            if _k != 1:
                # Needed to handle NaN case.
                assert max(
                    0, precision == (tp / (tp + fp))
                ), 'Metrics dont match.'

            # Create and save json
            results = {
                'ROC-AUC': roc_auc,
                'Precision-Recall': precision_recall_score,
                'MCC': mcc,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'Accuracy': accuracy,
                'Balanced Accuracy': balanced_accuracy,
                'Precision': precision,
                'theta': theta,
            }

            # Write data
            os.makedirs(os.path.join(results_path, f'k={_k}'), exist_ok=True)
            with open(
                os.path.join(
                    results_path, f'k={_k}', f'fold{fold}_report.json'
                ), 'w'
            ) as f:
                json.dump(results, f)
            all_results[idx].append(results)

            # Save predictions
            pd.DataFrame(
                {
                    'labels': test_labels,
                    'predictions': predictions,
                    'predicted_label': binary_predictions,
                }
            ).to_csv(
                os.path.join(
                    results_path, f'k={_k}', f'fold{fold}_results.csv'
                )
            )
        logger.info(f'Fold {fold}: Reports generated and saved.')

    # Generate reports across folds
    for idx, _k in enumerate(range(k, 0, -2)):
        df = pd.DataFrame(all_results[idx])
        df.index = range(num_folds)
        df.loc['mean'] = df.mean()
        df.loc['std'] = df.std()
        df.to_csv(os.path.join(results_path, f'knn_{_k}_cv_results.csv'))

    plot_roc_prc(results_path)
    logger.info('Done, shutting down.')


if __name__ == '__main__':
    args = parser.parse_args()
    main(
        args.data_path, args.train_name, args.test_name, args.num_folds,
        args.epitope_fp, args.tcr_fp, args.results_path, args.k,
        args.test_epitope_fp, args.test_tcr_fp
    )
