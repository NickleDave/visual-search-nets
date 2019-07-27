import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt


def train_history(acc_dir, set_sizes=(1, 2, 4, 8), save_as=None):
    """plots training history; specifically, accuracy on entire
     training set for each epoch, with separate lines on plot for each
    "set size" of visual search stimulus. Requires that the
    SAVE_ACC_BY_SET_SIZE_BY_EPOCH option in the config.ini file was
    set to True during training.

    Parameters
    ----------
    acc_dir : str
        path to directory where accuracy on training set, computed
        for each visual search stimulus set size, was saved during
        training
    set_sizes : tuple, list
        list of visual search stimulus set sizes.
        Default is (1, 2, 4, 8).
    save_as : str
        filename to save figure as. Default is None, in which case
        figure is not saved.

    Returns
    -------
    None
    """
    acc_txt_files = glob(os.path.join(acc_dir, '*.txt'))

    num_rows = len(acc_txt_files) / 3
    num_rows = int(np.ceil(num_rows))

    fig, ax = plt.subplots(num_rows, 3)
    fig.set_size_inches(15, 10)
    ax = ax.ravel()
    for ax_ind, acc_txt_file in enumerate(acc_txt_files):
        acc = np.loadtxt(acc_txt_file, delimiter=',')
        rows = acc.shape[0]
        for set_size, col in zip(set_sizes, acc.T):
            ax[ax_ind].plot(np.arange(rows), col, label=f"set_size: {set_size}")
        ax[ax_ind].set_title(f"replicate {ax_ind + 1}")
        ax[ax_ind].set_ylabel("acc")
        ax[ax_ind].set_xlabel("epoch")
        ax[ax_ind].legend(loc='lower right')

    if ax.shape[0] > len(acc_txt_files):
        extra = ax.shape[0] - len(acc_txt_files)
        for ind in range(1, extra + 1):
            ax[-ind].set_visible(False)

    fig.tight_layout()
    if save_as:
        plt.savefig(save_as)
