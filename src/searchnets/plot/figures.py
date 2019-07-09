import os
import json
from glob import glob

import joblib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns


mpl.style.use('seaborn-dark')

plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.labelweight'] = 'regular'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['axes.axisbelow'] = True


def acc_v_set_size(results, set_sizes=(1, 2, 4, 8), ax=None,
                   title=None, save_as=None, figsize=(10, 5),
                   set_xlabel=False, set_ylabel=False, set_ylim=True,
                   ylim=(0, 1.1), plot_mean=True, add_legend=False,
                   task_name=None):
    """plot accuracy as a function of visual search task set size
    for models trained on a single task or dataset

    Parameters
    ----------
    results
        path to results.gz file saved after measuring accuracy of trained networks
        on test set of visual search stimuli
    set_sizes : list
        of int, set sizes of visual search stimuli. Default is [1, 2, 4, 8].
    ax : matplotlib.Axis
        axis on which to plot figure. Default is None, in which case a new figure with
        a single axis is created for the plot.
    title : str
        string to use as title of figure. Default is None.
    save_as : str
        path to directory where figure should be saved. Default is None, in which
        case figure is not saved.
    figsize : tuple
        (width, height) in inches. Default is (10, 5). Only used if ax is None and a new
        figure is created.
    set_xlabel : bool
        if True, set the value of xlabel to "set size". Default is False.
    set_ylabel : bool
        if True, set the value of ylabel to "accuracy". Default is False.
    set_ylim : bool
        if True, set the y-axis limits to the value of ylim.
    ylim : tuple
        with two elements, limits for y-axis. Default is (0, 1.1).
    plot_mean : bool
        if True, find mean accuracy and plot as a separate solid line. Default is True.
    add_legend : bool
        if True, add legend to axis. Default is False.
    task_name : str


    Returns
    -------
    None
    """
    accs = joblib.load(results)['acc_per_set_size_per_model']
    accs = np.squeeze(accs)

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)

    for net_num, acc in enumerate(accs):
        label = f'net num. {net_num}'
        if task_name:
            label += ', task {task_name}'
        ax.plot(set_sizes, acc, linestyle='--', label=label)
    if plot_mean:
        mn_acc = accs.mean(axis=0)
        ax.plot(set_sizes, mn_acc, linewidth=3, label='mean', color='k')

    ax.set_xticks(set_sizes)

    if title:
        ax.set_title(title)
    if set_xlabel:
        ax.set_xlabel('set size')
    if set_ylabel:
        ax.set_ylabel('accuracy')
    if set_ylim:
        ax.set_ylim(ylim)

    if add_legend:
        ax.legend()

    if save_as:
        plt.savefig(save_as)


def metric_v_set_size_df(df, net_name, train_type, stimulus, metric, conditions,
                         ax=None, title=None, save_as=None, figsize=(10, 5),
                         both_color='darkgrey', mn_both_color='black',
                         set_xlabel=False, set_ylabel=False, set_ylim=True,
                         ylim=(0, 1.1), plot_mean=True, add_legend=False):
    """plot accuracy as a function of visual search task set size
    for models trained on a single task or dataset

    Accepts a Pandas dataframe and column names that determine what to plot.
    Dataframe is produces by searchstims.utils.general.results_csv function.

    Parameters
    ----------
    df : pandas.Dataframe
        path to results.gz file saved after measuring accuracy of trained networks
        on test set of visual search stimuli
    net_name : str
        name of neural net architecture. Must be a value in the 'net_name' column
        of df.
    train_type : str
        method used for training. Must be a value in the 'train_type' column of df.
    stimulus : str
        type of visual search stimulus, e.g. 'RVvGV', '2_v_5'. Must be a value in
        the 'stimulus' column of df.
    metric : str
        metric to plot. One of {'acc', 'd_prime'}.
    conditions : list, str
        conditions to plot. One of {'present', 'absent', 'both'}. Corresponds to
        'target_condition' column in df.

    Other Parameters
    ----------------
    ax : matplotlib.Axis
        axis on which to plot figure. Default is None, in which case a new figure with
        a single axis is created for the plot.
    title : str
        string to use as title of figure. Default is None.
    save_as : str
        path to directory where figure should be saved. Default is None, in which
        case figure is not saved.
    figsize : tuple
        (width, height) in inches. Default is (10, 5). Only used if ax is None and a new
        figure is created.
    both_color : str
        color to use to plot lines for individual replicates in 'both' condition
    mn_both_color : str
        color to use
    set_xlabel : bool
        if True, set the value of xlabel to "set size". Default is False.
    set_ylabel : bool
        if True, set the value of ylabel to metric. Default is False.
    set_ylim : bool
        if True, set the y-axis limits to the value of ylim.
    ylim : tuple
        with two elements, limits for y-axis. Default is (0, 1.1).
    plot_mean : bool
        if True, find mean accuracy and plot as a separate solid line. Default is True.
    add_legend : bool
        if True, add legend to axis. Default is False.

    Returns
    -------
    None
    """
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)

    df = df.loc[
        (df['net_name'] == net_name) & (df['train_type'] == train_type) &
        (df['stimulus'] == stimulus)]

    if 'present' in conditions:
        metric_target_present = []
    else:
        metric_target_present = None

    if 'absent' in conditions:
        metric_target_absent = []
    else:
        metric_target_absent = None

    if 'both' in conditions:
        metric_both = []
    else:
        metric_both = None

    set_sizes = None
    net_nums = df['net_number'].unique()
    # get metric across set sizes for each training replicate
    # we end up with a list of vectors we can pass to ax.plot,
    # so that the 'line' for each training replicate gets plotted
    for net_num in net_nums:
        df_this_net_num = df.loc[(df['net_number'] == net_num)]
        # and each condition specified
        for targ_cond in conditions:
            df_this_cond = df_this_net_num.loc[
                (df_this_net_num['target_condition'] == targ_cond)
            ]
            metric_vals = df_this_cond[metric].values
            if targ_cond == 'present':
                metric_target_present.append(metric_vals)
            elif targ_cond == 'absent':
                metric_target_absent.append(metric_vals)
            elif targ_cond == 'both':
                metric_both.append(metric_vals)

            set_size = df_this_cond['set_size'].values
            if set_sizes is None:
                set_sizes = set_size
            else:
                assert np.array_equal(set_sizes, set_size), 'set sizes did not match'

    if metric_target_present:
        for arr_metric_present in metric_target_present:
            ax.plot(set_sizes, arr_metric_present, color='violet', linewidth=2,
                    linestyle='--', marker='o', zorder=1, alpha=0.85, label=None)

        if plot_mean:
            mn_metric_present = np.asarray(metric_target_present).mean(axis=0)
            mn_metric_present_label = f'mean {metric},\ntarget present'
            mn_metric_present_line, = ax.plot(set_sizes, mn_metric_present,
                                              color='magenta', linewidth=4,
                                              zorder=0,
                                              label=mn_metric_present_label)
    else:
        mn_metric_present_line = None
        mn_metric_present_label = None

    if metric_target_absent:
        for arr_metric_absent in metric_target_present:
            ax.plot(set_sizes, arr_metric_absent, color='lightgreen', linewidth=2,
                    linestyle='--', marker='o', zorder=1, alpha=0.85, label=None)

        if plot_mean:
            mn_metric_absent = np.asarray(metric_target_absent).mean(axis=0)
            mn_metric_absent_label = f'mean {metric},\ntarget absent'
            mn_metric_absent_line, = ax.plot(set_sizes, mn_metric_absent,
                                             color='lawngreen', linewidth=4, zorder=0,
                                             label=mn_metric_absent_label)
    else:
        mn_metric_absent_line = None
        mn_metric_absent_label = None

    if metric_both:
        for arr_metric_both in metric_both:
            ax.plot(set_sizes, arr_metric_both, color=both_color, linewidth=2,
                    linestyle='--', marker='o', zorder=1, label=None)

        if plot_mean:
            mn_metric_both = np.asarray(metric_both).mean(axis=0)
            mn_metric_both_label = f'mean {metric}\n'
            mn_metric_both_line, = ax.plot(set_sizes, mn_metric_both, alpha=0.85,
                                             color=mn_both_color, linewidth=4, zorder=0,
                                             label=mn_metric_both_label)
    else:
        mn_metric_both_line = None
        mn_metric_both_label = None

    ax.set_xticks(set_sizes)

    if title:
        ax.set_title(title)
    if set_xlabel:
        ax.set_xlabel('set size')
    if set_ylabel:
        ax.set_ylabel(metric)
    if set_ylim:
        ax.set_ylim(ylim)

    if add_legend:
        handles = (handle for handle in [mn_metric_present_line,
                                         mn_metric_absent_line,
                                         mn_metric_both_line]
                   if handle is not None)
        labels = (label for label in [mn_metric_present_label,
                                      mn_metric_absent_label,
                                      mn_metric_both_label]
                  if label is not None)
        ax.legend(handles=handles,
                  labels=labels,
                  loc='lower left')

    if save_as:
        plt.savefig(save_as)


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
    figsize : tuple
        (width, height) in inches. Default is (10, 5).

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


def learncurve(csv_fname, figsize=(15, 10), ylim=(0, 0.5), suptitle=None, save_as=None):
    """"""
    df = pd.read_csv(csv_fname)
    set_sizes = sorted(df.set_size.unique())

    fig, ax = plt.subplots(1, len(set_sizes), figsize=figsize)
    for ax_ind, set_size in enumerate(set_sizes):
        df_this_ax = df.loc[df['set_size'] == set_size]
        sns.scatterplot(x='train_size', y='err', hue='setname', data=df_this_ax, legend=False, ax=ax[ax_ind])
        sns.lineplot(x='train_size', y='err', hue='setname', data=df_this_ax, ax=ax[ax_ind])
        ax[ax_ind].set_ylim(ylim)
        ax[ax_ind].set_xticks(df.train_size.unique())
        ax[ax_ind].set_xticklabels(df.train_size.unique(), rotation=45)
        ax[ax_ind].set_title(f'set size: {set_size}')
        if ax_ind == 0:
            ax[ax_ind].set_ylabel('error')
        else:
            ax[ax_ind].set_ylabel('')
        ax[ax_ind].set_xlabel('num. training samples')

    if suptitle:
        fig.suptitle(suptitle)

    if save_as:
        plt.savefig(save_as)
