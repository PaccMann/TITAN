import numpy as np


def cutoff_youdens_j(
    fpr: np.array, tpr: np.array, thresholds: np.array
) -> float:
    """Vanilla implementation of Youden's J-Score

    Args:
        fpr (np.array): Array of false positive rates (length N).
        tpr (np.array): Array of true positive rates (length N).
        thresholds ((np.array): Array of thresholds (length N).

    Returns:
        theta (float): threshold used for binarization of scores.
    """
    j_scores = tpr - fpr
    j_ordered = sorted(zip(j_scores, thresholds))
    return j_ordered[-1][1]
