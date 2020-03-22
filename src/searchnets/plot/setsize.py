"""plots of set size effects"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# default colors used for 'sampling units' (statistics term) in each condition
# in our case, a training replicate, one trained neural network
UNIT_COLORS = {
    'present': 'violet',
    'absent': 'lightgreen',
    'both': 'darkgrey'
}

# default colors used for plotting mean across sampling units in each condition
MN_COLORS = {
    'present': 'magenta',
    'absent': 'lawngreen',
    'both': 'black'
}


def metric_v_set_size_df(df, net_name, method, stimulus, metric, conditions,
                         unit_colors=UNIT_COLORS, mn_colors=MN_COLORS,
                         ax=None, title=None, save_as=None, figsize=(10, 5),
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
    method : str
        method used for training. Must be a value in the 'method' column of df.
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
    unit_colors : dict
        mapping of conditions to colors used for plotting 'sampling units', i.e. each trained
        network. Default is UNIT_COLORS defined in this module.
    mn_colors : dict
        mapping of conditions to colors used for plotting mean across 'sampling units'
        (i.e., each trained network). Default is MN_COLORS defined in this module.
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

    df = df[(df['net_name'] == net_name)
            & (df['method'] == method)
            & (df['stimulus'] == stimulus)]

    if not all(
            [df['target_condition'].isin([targ_cond]).any() for targ_cond in conditions]
    ):
        raise ValueError(f'not all target conditions specified were found in dataframe.'
                         f'Target conditions specified were: {conditions}')

    handles = []
    labels = []

    set_sizes = None  # because we verify set sizes is the same across conditions
    net_nums = df['net_number'].unique()
    # get metric across set sizes for each training replicate
    # we end up with a list of vectors we can pass to ax.plot,
    # so that the 'line' for each training replicate gets plotted
    for targ_cond in conditions:
        metric_vals = []
        for net_num in net_nums:
            metric_vals.append(
                df[(df['net_number'] == net_num)
                   & (df['target_condition'] == targ_cond)][metric].values
            )

            curr_set_size = df[(df['net_number'] == net_num)
                               & (df['target_condition'] == targ_cond)]['set_size'].values
            if set_sizes is None:
                set_sizes = curr_set_size
            else:
                if not np.array_equal(set_sizes, curr_set_size):
                    raise ValueError(
                        f'set size for net number {net_num}, '
                        f'target condition {targ_cond},  did not match others'
                    )

        for arr_metric in metric_vals:
            ax.plot(set_sizes, arr_metric, color=unit_colors[targ_cond], linewidth=2,
                    linestyle='--', marker='o', zorder=1, alpha=0.85, label=None)

        if plot_mean:
            mn_metric = np.asarray(metric_vals).mean(axis=0)
            if targ_cond == 'both':
                mn_metric_label = f'mean {metric}, all trials, {method}'
            else:
                mn_metric_label = f'mean {metric}, {targ_cond}, {method}'
            labels.append(mn_metric_label)
            mn_metric_line, = ax.plot(set_sizes, mn_metric,
                                      color=mn_colors[targ_cond], linewidth=4,
                                      zorder=0,
                                      label=mn_metric_label)
            handles.append(mn_metric_line)

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
        ax.legend(handles=handles,
                  labels=labels,
                  loc='lower left')

    if save_as:
        plt.savefig(save_as)


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
