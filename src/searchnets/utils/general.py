import os
from pathlib import Path
import tarfile

import joblib
import numpy as np
import pandas as pd

from .metrics import compute_d_prime


def make_save_path(save_path, net_name, net_number, epochs):
    """make a unique save path for model and checkpoints,
     using network architecture, training replicate number, and number of epochs"""
    save_path = Path(save_path).joinpath(
        f'trained_{epochs}_epochs',
        f'net_number_{net_number}')
    if not save_path.is_dir():
        save_path.mkdir(parents=True, exist_ok=True)
    stem = f'{net_name}_trained_{epochs}_epochs_number_{net_number}'
    save_path = save_path.joinpath(stem)
    return save_path


def make_targz(output_filename, source_dir):
    """make a .tar.gz file from a directory, i.e. a gzip-compressed archive"""
    with tarfile.open(name=output_filename, mode="w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def targz_dirs(dir_names=('checkpoints', 'configs', 'data_prepd_for_nets', 'results', 'visual_search_stimuli'),
               path='.'):
    """makes a .tar.gz archive for a list of directories, for uploading to figshare.

    Parameters
    ----------
    dir_names : list
        of directory names.
        Default is ['checkpoints', 'configs', 'data_prepd_for_nets', 'results', 'visual_search_stimuli'].
    path : str
        Path to parent directory where all the directories in dir_names can be found.
        Default is '.', i.e. current working directory.

    Notes
    -----
    Basically just a wrapper around make_targz for convenience;
    assumes each .tar.gz output should have the name 'directory_name.tar.gz'
    """
    for dir_name in dir_names:
        output_filename = dir_name + '.tar.gz'
        print(f'making {output_filename}')
        make_targz(output_filename=output_filename, source_dir=os.path.join(path, dir_name))


COLUMNS = ['net_name',
           'method',
           'learning_rate',
           'net_number',
           'stimulus',
           'set_size',
           'target_condition',
           'accuracy',
           'hit_rate',
           'false_alarm_rate',
           'd_prime',
           ]


def results_df(data_csv_path,
               results_gz_path,
               net_name,
               method,
               learning_rate,
               results_csv_path=None,
               ):
    """creates Pandas dataframe from results.
    The resulting dataframe can be used with searchstims.plot.figures.metric_v_set_size_df

    Parameters
    ----------
    data_csv_path : str
        path to .csv file of dataset splits, created by running 'searchnets split' from the command line.
    results_gz_path
        paths to .gz file created by running 'searchnets test' from the command line.
    net_name : str
        name of convolutional neural net architecture to train.
        One of {'alexnet', 'VGG16'}
    method : str
        training method. One of {'initialize', 'transfer'}. See docstring for searchnets.train.
    learning_rate : float
        hyperparameter used during weight update step of training network.
    results_csv_path : str
        Path to use to save dataframe as a csv.
        Default is None, in which case no csv is saved.

    Returns
    -------
    df : pandas.Dataframe
        computed from results
    """
    df_dataset = pd.read_csv(data_csv_path)
    df_testset = df_dataset[df_dataset['split'] == 'test']
    # add y_true column to df_testset that we compare to y_pred
    # (can't compare y_pred to target_condition column since that is text,
    # not integer like output of network)
    df_testset['y_true'] = df_testset['target_condition'] == 'present'
    set_sizes = df_testset['set_size'].unique()

    stims = df_testset['stimulus'].unique().tolist()

    results_dict = joblib.load(results_gz_path)

    rows = []

    preds_per_model = results_dict['predictions_per_model_dict']
    num_nets = len(preds_per_model.keys())
    for net_num in range(1, num_nets + 1):
        # below, notice we need to convert key (a Path) to string to check if net_num is 'in' it
        key = [key for key in preds_per_model.keys() if f'net_number_{net_num}' in str(key)][0]
        # add y_pred column to test set dataframe
        # notice we overwrite this column every time through the loop, i.e. for each trained network
        df_testset['y_pred'] = preds_per_model[key]

        for stim in stims:
            stim_df = df_testset[df_testset['stimulus'] == stim]

            for target_cond in ('present', 'absent', 'both'):
                for set_size in set_sizes:
                    set_size_df = stim_df[stim_df['set_size'] == set_size]
                    if target_cond == 'present':
                        cond_df = set_size_df[set_size_df['y_true'] == 1]
                    elif target_cond == 'absent':
                        cond_df = set_size_df[set_size_df['y_true'] == 0]
                    elif target_cond == 'both':
                        cond_df = set_size_df
                    correct_bool = cond_df['y_true'] == cond_df['y_pred']
                    import pdb;pdb.set_trace()
                    acc = np.sum(correct_bool) / correct_bool.shape[0]

                    if target_cond == 'both':
                        hit_rate, false_alarm_rate, d_prime = compute_d_prime(
                            cond_df['target_condition'].to_numpy(), cond_df['y_pred'].to_numpy()
                        )
                    else:
                        hit_rate, false_alarm_rate, d_prime = None, None, None
                    row = [net_name, method, learning_rate, net_num, stim, set_size, target_cond,
                           acc, hit_rate, false_alarm_rate, d_prime]
                    rows.append(row)

    df = pd.DataFrame.from_records(rows, columns=COLUMNS)
    if results_csv_path:
        df.to_csv(results_csv_path)
    return df


def reorder_paths(paths, order_strs):
    """reorder a list of paths, using a list of strings.
    Returns a new list of the paths, re-ordered so that the
    first path will have the first string in it, the second path
    will have the second string in it, and so on.

    Parameters
    ----------
    paths : list
        of paths
    order_strs : list
        of strings, e.g. visual search stimulus names

    Returns
    -------
    paths_out : list
        paths, sorted by order_strs

    Notes
    -----
    Used to sort paths to data and results, according to
    visual search stimulus names
    """
    if len(paths) != len(order_strs):
        raise ValueError(
            "length of paths does not equal length of order_strs"
        )

    paths_out = []
    for order_str in order_strs:
        for path in paths:
            if order_str in path:
                paths_out.append(path)

    assert len(paths_out) == len(paths), "not all paths in paths_out"

    return paths_out
