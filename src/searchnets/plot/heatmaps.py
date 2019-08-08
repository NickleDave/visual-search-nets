import json
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt

from ..utils.metrics import p_item_grid, acc_grid, err_grid


def heatmap(grid, ax=None, cmap='rainbow', vmin=0, vmax=1):
    """helper function that plots a heatmap,
     using the matplotlib.pyplot.imshow function

    Parameters
    ----------
    grid : numpy.ndarray
        2-dimensional array to plot as a heatmap
    ax : matplotlib.axes.Axes
        axes on which to plot heatmap. If None, then
        a new Axes instance is created
    cmap : str
        name of colormap to use. Default is 'rainbow'.
    vmin, vmax : scalar
        define the data range that the colormap covers.
        Default is 0, 1.

    Returns
    -------
    im : matplotlib.image.AxesImage
        instance returned by call to imshow.
    ax : matplotlib.axes.Axes
        axes on which heatmap was plotted.
    """
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(grid, vmin=vmin, vmax=vmax, cmap=cmap)
    return im, ax


def p_item_heatmap(json_fname, data_gz_fname, stim_abbrev, set_size=None,
                   data_set='train', item_char='t', vmin=0, vmax=1,
                   ax=None, cmap='rainbow'):
    """given a dataset of visual search stimuli where discrete target and
    distractor items appear within cells of a grid, plot a
    heatmap of the probability that a specified item appears
    within each cell of the grid

    Parameters
    ----------
    json_fname : str
        path to .json file created by searchstims package
        with target and distractor indices from each stimulus file
    data_gz_fname : str
        path to .gz file created by searchnets.data
        that has filenames of files in dataset
    stim_abbrev : str
        abbreviation that represents type of visual search stimulus,
        e.g. '2_v_5', 'RVvGV'.
    set_size : int
        set size of visual search stimuli
    data_set : str
        one of {'train', 'val', 'test'}; data set from which heatmap should
        be generated. Default is 'train'.
    item_char : str
        Character that represents item in grid_as_char.
        Default is 't', for 'target'.
    ax : matplotlib.axes.Axes
        default is None, in which case a new Axes instance is created.
    cmap : str
        name of colormap to use. Default is 'rainbow'.

    Returns
    -------
    p : numpy.ndarray
        where value of each element is probability
        that item_char occurs in the corresponding cell in char_grids
    im : matplotlib.image.AxesImage
        instance returned by call to imshow
    ax : matplotlib.axes.Axes
        axes on which heatmap was plotted.
    """
    if data_set not in {'train',  'val', 'test'}:
        raise ValueError(
            f"data_set must be 'train', 'test', or 'val', but got {data_set}"
        )

    # get filenames of files in dataset, i.e. training set files, test files, etc.,
    # so we can keep only grid_as_char for those files
    data_gz = joblib.load(data_gz_fname)
    stim_fnames = data_gz[f'x_{data_set}']

    if all([type(stim_fname) == list for stim_fname in stim_fnames]):
        stim_fnames = [item for sublist in stim_fnames for item in sublist]

    if all([type(stim_fname) == str for stim_fname in stim_fnames]):
        stim_fnames = np.asarray(stim_fnames)
    elif all([type(stim_fname) == np.ndarray for stim_fname in stim_fnames]):
        stim_fnames = np.concatenate(stim_fnames)

#    stim_type_vec = data_gz[f'stim_type_vec_{data_set}']
#    if all([type(stim_fname) == str for stim_fname in stim_fnames]):
#        stim_fnames = np.asarray(stim_fnames)
#    elif all([type(stim_fname) == np.ndarray for stim_fname in stim_fnames]):
#        stim_fnames = np.concatenate(stim_fnames)

    # keep just the ones that are the correct visual search stimulus type
#    inds_of_stim = np.nonzero(stim_type_vec == stim_abbrev)

#    stim_fnames = stim_fnames[inds_of_stim]
#    stim_fnames = stim_fnames.tolist()
    # keep just name of file, not whole path,
    # so we can more easily check if each filename from the .json file
    # is in this list of 'just filenames' (without paths)
    stim_fnames = [Path(stim_fname).name for stim_fname in stim_fnames]

    with open(json_fname) as fp:
        stim_meta_dict = json.load(fp)

    stim_meta_dict = stim_meta_dict[stim_abbrev]
    # --- notice we filter by set size here so we don't need to do it above, we will only get metadata for the
    # set size we want ---
    stim_meta_dict = stim_meta_dict[set_size]
    # just concatenate both the 'present' and 'absent' lists
    # and look at all of them, instead of e.g. only keeping
    # target present to look for 't'. In case a different
    # character was used for target and/or distractor type(s)
    stim_meta_list = []
    stim_meta_list.extend(stim_meta_dict['present'])
    stim_meta_list.extend(stim_meta_dict['absent'])

    char_grids = [
        np.asarray(meta_d['grid_as_char'])
        for meta_d in stim_meta_list
        # only keep if filename is in stim_fnames from data_gz
        if Path(meta_d['filename']).name in stim_fnames
    ]

    p = p_item_grid(char_grids, item_char)
    im, ax = heatmap(p, ax, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    return p, im, ax


def acc_err_heatmap(json_fname, data_gz_fname, results_gz_fname,
                    stim_abbrev, net_num, set_size=None,
                    metric='acc', data_set='test',
                    vmin=0, vmax=1, ax=None, cmap='rainbow'):
    """given a dataset of visual search stimuli where discrete target and
    distractor items appear within cells of a grid, plot heatmap of
    either accuracy or error within each cell of the grid,
    where accuracy = number of correct trials / total number of trials with an item in this cell
    and error = 1 - accuracy

    Parameters
    ----------
    json_fname : str
        path to .json file created by searchstims package
        with target and distractor indices from each stimulus file
    data_gz_fname : str
        path to .gz file created by searchnets.data
        that has filenames of files in dataset
    results_gz_fname : str
        path to .gz file created by searchnets.test
        that has predictions output by trained network
    stim_abbrev : str
        abbreviation that represents type of visual search stimulus,
        e.g. '2_v_5', 'RVvGV'.
    net_num : int
        which training replicate to use
    set_size : int
        set size of visual search stimuli
    metric : str
        one of {'acc', 'err'). Default is 'acc'.
    data_set : str
        one of {'train', 'test'}; data set from which heatmap should
        be generated. Default is 'test'.
    item_char : str
        Character that represents item in grid_as_char.
        Default is 't', for 'target'.
    ax : matplotlib.axes.Axes
        default is None, in which case a new Axes instance is created.
    cmap : str
        name of colormap to use. Default is 'rainbow'.

    Returns
    -------
    m_arr : numpy.ndarray
        of computed metric, accuracy or error
        where value of each element is probability
        that item_char occurs in the corresponding cell in char_grids
    im : matplotlib.image.AxesImage
        instance returned by call to imshow
    ax : matplotlib.axes.Axes
        axes on which heatmap was plotted.
    """
    if data_set not in {'train',  'test'}:
        raise ValueError(
            f"data_set must be 'train' or 'test', but got {data_set}"
        )

    if metric not in {'acc', 'err'}:
        raise ValueError(
            f"metric must be 'acc' or 'err', but was '{metric}'"
        )

    if type(set_size) != int:
        raise TypeError(
            f'type of set size should be int, not {type(set_size)}. Function converts to string where required.'
        )

    # get filenames of files in dataset, i.e. training set files, test files, etc.,
    # so we can keep only grid_as_char for those files
    data_gz = joblib.load(data_gz_fname)
    stim_fnames = data_gz[f'x_{data_set}']

    if all([type(stim_fname) == list for stim_fname in stim_fnames]):
        stim_fnames = [item for sublist in stim_fnames for item in sublist]

    if all([type(stim_fname) == str for stim_fname in stim_fnames]):
        stim_fnames = np.asarray(stim_fnames)
    elif all([type(stim_fname) == np.ndarray for stim_fname in stim_fnames]):
        stim_fnames = np.concatenate(stim_fnames)
    y_true = data_gz[f'y_{data_set}']

    # --- unlike in p_item_heatmap function, we filter by set size here as well as below, because
    # we need this list to match the list we get out of the metadata file exactly so we can figure out
    # which entry in y_true from data_gz and y_pred from results_gz corresponds to
    # a grid_as_char from the metadata file that tells us where the items were in the visual search stimulus
    set_size_vec = data_gz[f'set_size_vec_{data_set}']
    inds_this_set_size = np.nonzero(set_size_vec == set_size)[0]
    stim_fnames = stim_fnames[inds_this_set_size]
    y_true = y_true[inds_this_set_size]

    # keep just name of file, not whole path,
    # so we can more easily check if each filename from the .json file
    # is in this list of 'just filenames' (without paths)
    stim_fnames = [Path(stim_fname).name for stim_fname in stim_fnames]

    with open(json_fname) as fp:
        stim_meta_dict = json.load(fp)

    stim_meta_dict = stim_meta_dict[stim_abbrev]
    # --- in line below, suddenly convert set_size to string because that's how it is in dictionary
    # but need set_size to be an int everywhere else ---
    stim_meta_dict = stim_meta_dict[str(set_size)]
    # just concatenate both the 'present' and 'absent' lists
    # and look at all of them, instead of e.g. only keeping
    # target present to look for 't'. In case a different
    # character was used for target and/or distractor type(s)
    stim_meta_list = []
    stim_meta_list.extend(stim_meta_dict['present'])
    stim_meta_list.extend(stim_meta_dict['absent'])

    char_grids = []
    stim_fnames_meta = []
    for meta_d in stim_meta_list:
        # only keep if filename is in stim_fnames from data_gz
        stim_fname_meta = Path(meta_d['filename']).name
        if stim_fname_meta in stim_fnames:
            char_grids.append(np.asarray(meta_d['grid_as_char']))
            stim_fnames_meta.append(stim_fname_meta)

    results_gz = joblib.load(results_gz_fname)
    preds = results_gz['predictions_per_model_dict']
    pred_key = [k for k in preds.keys() if f'net_number_{net_num}' in k]
    if len(pred_key) != 1:
        raise ValueError(
            'Did not find just one key that matched net number in dictionary of predictions.\n'
            f'Keys found were: {pred_key}'
        )
    pred_key = pred_key[0]
    y_pred = preds[pred_key]
    # filter y_pred by set_size as well
    y_pred = y_pred[inds_this_set_size]

    if metric == 'acc':
        m_arr = acc_grid(stim_fnames, y_true, y_pred, char_grids, stim_fnames_meta)
    elif metric == 'err':
        m_arr = err_grid(stim_fnames, y_true, y_pred, char_grids, stim_fnames_meta)
    im, ax = heatmap(m_arr, ax, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    return m_arr, im, ax
