from typing import Iterable, List

import numpy as np
from Levenshtein import distance as levenshtein_distance


def knn(
    train_epitopes: List[str],
    train_tcrs: List[str],
    train_labels: List[int],
    test_epitopes: List[str],
    test_tcrs: List[str],
    k: int = 1,
    return_knn_labels: bool = False,
):
    """Baseline model for epitope-TCR binding prediction. Applies KNN classification
    using as similarity the length-normalized Levensthein distance between epitopes and
    TCRs. Predictions conceptually correspond to the predict_proba method of
    sklearn.neighbors.KNeighborsClassifier.

    Args:
        train_epitopes (List[str]): List of AA sequences of training epitope samples.
            Length should be identical to train_tcrs and train_labels.
        train_tcrs (List[str]): List of AA sequences of training TCR samples. Length
            should be identical to train_epitopes and train_labels.
        train_labels (List[int]): List of training labels. Length should be identical
            to train_tcrs and train_labels.
        test_epitopes (List[str]): List of AA sequences of test epitope samples. Length
            should be identical to test_tcrs.
        test_tcrs (List[str]): List of AA sequences of test TCR samples. Length should
            be identical to test_epitopes
        k (int, optional): Hyperparameter for KNN classification. Defaults to 1.
        return_knn_labels (bool, optional): If set, the labels of the K nearest
            neighbors are also returned.
    """
    assert isinstance(train_epitopes, Iterable)
    assert isinstance(train_tcrs, Iterable)
    assert isinstance(train_labels, Iterable)
    assert isinstance(test_epitopes, Iterable)
    assert isinstance(test_tcrs, Iterable)

    assert len(test_tcrs) == len(test_epitopes), 'Test data lengths dont match'
    assert len(train_epitopes
               ) == len(train_tcrs), 'Test data lengths dont match'
    assert len(train_epitopes
               ) == len(train_labels), 'Test data lengths dont match'

    predictions, knn_labels = [], []
    for epitope, tcr in zip(test_epitopes, test_tcrs):

        el = len(epitope)
        tl = len(tcr)
        epitope_dists = [
            levenshtein_distance(epitope, e) / el for e in train_epitopes
        ]
        tcr_dists = [levenshtein_distance(tcr, t) / tl for t in train_tcrs]

        knns = np.argsort(np.array(epitope_dists) + np.array(tcr_dists))[:k]
        _knn_labels = np.array(train_labels)[knns]
        predictions.append(np.mean(_knn_labels))
        knn_labels.append(_knn_labels)

    return (predictions, knn_labels) if return_knn_labels else predictions
