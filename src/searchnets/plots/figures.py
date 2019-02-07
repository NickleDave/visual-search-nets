import os
import json

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('bmh')

plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.labelweight'] = 'regular'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 20


def plot_results(eff_accs, ineff_accs, epochs,
                 set_sizes=[1, 2, 4, 8], savefig=False, savedir=None):
    """plot accuracy of trained models on visual search task
    with separate plots for efficient and inefficient search stimuli

    Parameters
    ----------
    eff_accs
    ineff_accs
    epochs
    set_sizes
    savefig
    savedir

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(10, 5)
    ax = ax.ravel()

    ax[0].plot(SET_SIZES, eff_accs.T)
    ax[0].set_xticks(SET_SIZES)
    ax[0].set_title('efficient')
    ax[0].set_xlabel('set size')
    ax[0].set_ylabel('accuracy')

    ax[1].plot(SET_SIZES, ineff_accs.T)
    ax[1].set_xticks(SET_SIZES)
    ax[1].set_title('inefficient')
    ax[1].set_xlabel('set size')
    ax[1].set_ylim([0, 1.1])

    plt.tight_layout()

    if savefig:
        fname = os.path.join(savedir, f'alexnet_eff_v_ineff_{epochs}_epochs.png')
        plt.savefig(fname)


def item_heatmap(json_fname, data_npz_fname, test_npz_fname,
                 item_type, pred_type):
    """create heatmap of where items were plotted
    to visualize whether location of items affected accuracy

    Parameters
    ----------
    json_fname : str
        path to .json file output by searchstims with target
        and distractor indices from each stimulus file
    data_npz_fname : str
        path to .npz file output by searchnets.data that
        contains list of filenames for training, validation, and
        test sets. Used to link predictions to data in json_fname.
    test_npz_fname : str
        path to .npz file output by searchnets.test that
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

    with np.load(data_npz_fname) as npz:
        test_set_files = npz['test_set_files']
        y_true = npz['y_test']
        set_size_vec_test = npz['set_size_vec_test']

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

    with np.load(test_npz_fname) as npz:
        predictions_per_model_dict = npz['predictions_per_model_dict']

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
        import pdb;pdb.set_trace()
        these_inds = np.asarray(these_inds)
        xx.append(these_inds[:, 0])
        yy.append(these_inds[:, 1])

    fig, ax = plt.subplots()
    for x_vals, y_vals in zip(xx, yy):
        ax.plot(x_vals, y_vals)
