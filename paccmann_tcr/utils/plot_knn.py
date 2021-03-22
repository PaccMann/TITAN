import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_roc_prc(path: str) -> None:
    """
    Aggregates CV results for different choices of k and creates a plot with ROC-AUC
    and PRC curve. Also saves df with mean/stds

    Args:
        path (str): Path to a folder with files called `knn_{k}_cv_results.csv` with
            differrent choices of k (e.g. generated with knn_baseline.py)
    """

    dfs = []
    for filepath in glob.iglob(path + '/*.csv'):
        k = int(filepath.split('/')[-1].split('_')[1])
        df = pd.read_csv(filepath)
        df.rename(columns={'Unnamed: 0': 'fold'}, inplace=True)
        df['k'] = k
        dfs.append(df)
    df = pd.concat(dfs)
    df.index = range(len(df))
    mean_df = df[df['fold'].isin(['mean', 'std'])]
    mean_df = mean_df.sort_values(
        by=['fold', 'ROC-AUC'], ascending=[True, False]
    )
    df = df[~df['fold'].isin(['mean', 'std'])]

    fig, (ax1, ax2
          ) = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(20, 5))
    plt.suptitle('KNN - VDJ Validation performance', size=15)
    sns.boxenplot(data=df, x='k', y='ROC-AUC', ax=ax1)
    ax1.set_title('Area under ROC')
    sns.boxenplot(data=df, x='k', y='Precision-Recall', ax=ax2)
    ax2.set_title('Area under Precision Recall Curve')
    plt.savefig(os.path.join(path, 'roc_pc_curves.pdf'))
    mean_df.to_csv(os.path.join(path, 'summary.csv'))
