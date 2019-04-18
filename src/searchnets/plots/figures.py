import os
import json
from glob import glob

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import joblib
from scipy import stats

mpl.style.use('bmh')

plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.labelweight'] = 'regular'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 20


def ftr_v_spt_conj(ftr_results, spt_conj_results, epochs,
                set_sizes=(1, 2, 4, 8), savefig=False, savedir=None,
                figsize=(10, 5)):
    """plot accuracy of trained models on visual search task
    with separate plots for feature and spatial conjunction search stimuli

    Parameters
    ----------
    ftr_results : str
        path to results.gz file saved after measuring accuracy of trained convnets
        on test set of feature search stimuli
    spt_conj_results
        path to results.gz file saved after measuring accuracy of trained convnets
        on test set of feature search stimuli
    epochs : int
        number of epochs that nets were trained
    set_sizes : list
        of int, set sizes of visual search stimuli. Default is [1, 2, 4, 8].
    savefig : bool
        if True, save figure. Default is False.
    savedir : str
        path to directory where figure should be saved. Default is None.

    Returns
    -------
    None
    """
    ftr_accs = joblib.load(ftr_results)['acc_per_set_size_per_model']
    ftr_accs = np.squeeze(ftr_accs)
    spt_conj_accs = joblib.load(spt_conj_results)['acc_per_set_size_per_model']
    spt_conj_accs = np.squeeze(spt_conj_accs)

    fig, ax = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(figsize)
    ax = ax.ravel()

    ax[0].plot(set_sizes, ftr_accs.T)
    ax[0].set_xticks(set_sizes)
    ax[0].set_title('feature')
    ax[0].set_xlabel('set size')
    ax[0].set_ylabel('accuracy')

    ax[1].plot(set_sizes, spt_conj_accs.T)
    ax[1].set_xticks(set_sizes)
    ax[1].set_title('spatial conjunction')
    ax[1].set_xlabel('set size')
    ax[1].set_ylim([0, 1.1])

    fig.suptitle(f'{epochs} epochs')

    if savefig:
        fname = os.path.join(savedir, f'alexnet_ftr_v_spt_conj_{epochs}_epochs.png')
        plt.savefig(fname)


def mn_slope_by_epoch(ftr_results_list, spt_conj_results_list, epochs_list,
                      set_sizes=(1, 2, 4, 8), savefig=False, savedir=None,
                      figsize=(20, 5)):
    """plot accuracy as a function of number of epochs of training

    Parameters
    ----------
    ftr_results_list
    spt_conj_results_list
    epochs_list

    Returns
    -------
    None
    """
    ftr_slopes = []
    spt_conj_slopes = []
    for ftr_results, spt_conj_results, epochs in zip(ftr_results_list, spt_conj_results_list, epochs_list):
        ftr_accs = joblib.load(ftr_results)['acc_per_set_size_per_model']
        ftr_accs = np.squeeze(ftr_accs)
        ftr_slopes_this_epochs = []
        for acc_row in ftr_accs:
            slope, intercept, r_value, p_value, std_err = stats.linregress(set_sizes, acc_row)
            ftr_slopes_this_epochs.append(slope)
        ftr_slopes_this_epochs = np.asarray(ftr_slopes_this_epochs)
        ftr_slopes.append(ftr_slopes_this_epochs)

        spt_conj_accs = joblib.load(spt_conj_results)['acc_per_set_size_per_model']
        spt_conj_accs = np.squeeze(spt_conj_accs)
        spt_conj_slopes_this_epochs = []
        for acc_row in spt_conj_accs:
            slope, intercept, r_value, p_value, std_err = stats.linregress(set_sizes, acc_row)
            spt_conj_slopes_this_epochs.append(slope)
        spt_conj_slopes_this_epochs = np.asarray(spt_conj_slopes_this_epochs)
        spt_conj_slopes.append(spt_conj_slopes_this_epochs)

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(figsize)

    bpl = ax[0].boxplot(ftr_slopes, sym='', widths=0.6)
    ax[0].set_xticklabels(epochs_list)
    ax[0].set_ylabel('slope')
    ax[0].set_ylim([-0.1, 0.])
    ax[0].set_xlabel('number of\ntraining epochs')
    ax[0].set_title('feature')
    set_box_color(bpl, '#D7191C')  # colors are from http://colorbrewer2.org/

    bpr = ax[1].boxplot(spt_conj_slopes, sym='', widths=0.6)
    ax[1].set_xticklabels(epochs_list)
    ax[1].set_ylabel('slope')
    ax[1].set_ylim([-0.1, 0.])
    ax[1].set_xlabel('number of\ntraining epochs')
    ax[1].set_title('spatial conjunction')
    set_box_color(bpr, '#2C7BB6')

    mn_ftr_slopes = np.asarray([np.mean(slopes) for slopes in ftr_slopes])
    mn_spt_conj_slopes = np.asarray([np.mean(slopes) for slopes in spt_conj_slopes])
    diffs = mn_ftr_slopes - mn_spt_conj_slopes

    ax[2].bar(range(len(epochs_list)), diffs)
    ax[2].set_xticks(range(len(epochs_list)))
    ax[2].set_xticklabels(epochs_list)
    ax[1].set_title('spatial conjunction')
    ax[2].set_ylabel('slope difference\n(feature - spatial conjunction)')
    ax[2].set_xlabel('number of\ntraining epochs')
    ax[2].set_title('difference')

    plt.tight_layout()
    if savefig:
        plt.savefig('boxcompare.png')


def item_heatmap(json_fname, data_fname, test_results_fname,
                 item_type, pred_type):
    """create heatmap of where items were plotted
    to visualize whether location of items affected accuracy

    Parameters
    ----------
    json_fname : str
        path to .json file output by searchstims with target
        and distractor indices from each stimulus file
    data_fname : str
        path to file output by searchnets.data that
        contains list of filenames for training, validation, and
        test sets. Used to link predictions to data in json_fname.
    test_results_fname : str
        path to file output by searchnets.test that
        contains predictions for each trained neural network.
    item_type : str
        one of {'target', 'distractor'}
    pred_type : str
        one of {'correct', 'incorrect'}

    Returns
    -------
    None
    """
    if item_type not in ['target', 'distractor']:
        raise ValueError("pred_type must be either 'target' or 'distractor'")

    if pred_type not in ['correct', 'incorrect']:
        raise ValueError("pred_type must be either 'correct' or 'incorrect'")

    with open(json_fname) as fp:
        stim_info_json = json.load(fp)

    # get indices out of .json file
    if item_type == 'target':
        inds_key = 'target_indices'
    elif item_type == 'distractor':
        inds_key = 'distractor_indices'
    item_indices_by_fname = {
        os.path.basename(stim_info['filename']): stim_info[inds_key]
        for set_size, present_absent_dict in stim_info_json.items()
        for is_target_present, stim_info_list in present_absent_dict.items()
        for stim_info in stim_info_list
    }

    data = joblib.load(data_fname)
    test_set_files = data['test_set_files']
    y_true = data['y_test']
    set_size_vec_test = data['set_size_vec_test']

    # get indices using file names from
    # 'test_set_files' in .npz file with training data,
    # so indices are in same order as y_test from the same file
    y_true_item_indices = []
    for test_set_file in test_set_files:
        y_true_item_indices.append(
            item_indices_by_fname[os.path.basename(test_set_file)]
        )

    # now convert to a numpy array so we can index into it;
    # note this is an "array of lists", because one image may have multiple
    # sets of indices for distractors, for example, but I still want to be able to
    # index by image without doing a bunch of advanced indexing
    y_true_item_indices = np.asarray(y_true_item_indices).squeeze()

    results = joblib.load(test_results_fname)
    predictions_per_model_dict = results['predictions_per_model_dict']

    y_pred = list(predictions_per_model_dict.values())

    to_plot = []
    for y_pred_for_model in y_pred:
        if pred_type == 'correct':
            pred_indices = np.where(y_pred_for_model == y_true)[0]
        elif pred_type == 'incorrect':
            pred_indices = np.where(y_pred_for_model != y_true)[0]
        to_plot.append(pred_indices)

    xx = []
    yy = []
    for pred_indices in to_plot:
        these_rows = y_true_item_indices[pred_indices]
        # have to flatten lists of pairs of indices into one long list of pairs of indices
        these_inds = []  # <-- will be long list
        for row in these_rows:
            for inds_pair in row:
                these_inds.append(inds_pair)
        these_inds = np.asarray(these_inds)
        xx.append(these_inds[:, 0])
        yy.append(these_inds[:, 1])

    fig, ax = plt.subplots()
    for x_vals, y_vals in zip(xx, yy):
        ax.scatter(x_vals, y_vals)


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
