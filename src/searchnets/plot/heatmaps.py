import json
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt

from ..utils.metrics import p_item_grid


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
    """
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(grid, vmin=vmin, vmax=vmax, cmap=cmap)
    return im


def p_item_heatmap(json_fname, data_gz_fname, stim_abbrev, set_size=None,
                   data_set='train', item_char='t', vmin=0, vmax=1,
                   ax=None, add_cbar=False):
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

    Returns
    -------
    p : numpy.ndarray
        where value of each element is probability
        that item_char occurs in the corresponding cell in char_grids
    im : matplotlib.image.AxesImage
        instance returned by call to imshow
    """
    if data_set not in {'train',  'val', 'test'}:
        raise ValueError(
            f"data_set must be 'train', 'test', or 'val', but got {data_set}"
        )

    # get filenames of files in dataset, i.e. training set files, test files, etc.,
    # so we can keep only grid_as_char for those files
    data_gz = joblib.load(data_gz_fname)
    stim_fnames = data_gz[f'x_{data_set}']
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
#    import pdb;pdb.set_trace()
#    stim_fnames = stim_fnames[inds_of_stim]
#    stim_fnames = stim_fnames.tolist()
    # keep just name of file, not whole path,
    # so we can more easily check if each filename from the .json file
    # is in this list of 'just filenames' (without paths)
    stim_fnames = [Path(stim_fname).name for stim_fname in stim_fnames]

    with open(json_fname) as fp:
        stim_meta_dict = json.load(fp)

    stim_meta_dict = stim_meta_dict[stim_abbrev]
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
    im = heatmap(p, ax, vmin=vmin, vmax=vmax)
    return p, im