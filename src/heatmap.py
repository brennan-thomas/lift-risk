import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import os

def heatmap(mat, title, file):
    dir = os.path.dirname(file)
    if len(dir) > 0:
        os.makedirs(dir, exist_ok=True)

    mat = np.array(mat)
    norm_rows = (mat.T / mat.sum(axis=1)).T

    sns.set()

    f,ax = plt.subplots(1, 1, figsize=(5,5), dpi=300)

    mat_sum = mat.sum(axis = 1)
    def get_acc(i):
        return str(round(mat[i,i] / mat_sum[i], 2))
    annot = np.diag([get_acc(i) for i in range(4)])

    total_accuracy = sum([mat[i, i] for i in range(4)]) / mat_sum.sum()

    sns.heatmap(norm_rows, ax=ax, fmt='', cmap=plt.cm.get_cmap('viridis'), annot=annot)
    ax.set_title(title)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_xticklabels(['Low', 'Medium', 'High', 'None'])
    ax.set_yticklabels(['Low', 'Medium', 'High', 'None'])
    f.tight_layout()

    plt.figtext(0.99, 0.01, 'Total Accuracy: {:.2f}'.format(total_accuracy), horizontalalignment='right')
    plt.savefig(file)
    plt.clf()
