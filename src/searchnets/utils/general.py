import tarfile
import os

import joblib
import numpy as np
import pandas as pd


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


HEADER = ['net_name', 'train_type', 'net_number', 'stimulus', 'target_condition', 'set_size', 'accuracy']


def results_csv(data_prep_dir,
                results_dir,
                test_csv_path='./test.csv',
                nets=('alexnet', 'VGG16'),
                train_types=('finetune', 'train'),
                stims=('2_v_5', 'RVvGV', 'RVvRHGV'),
                target_condition=('present', 'absent'),
                data_gz_paths=None,
                results_gz_paths=None,
                ):
    """make csv from results directory

    creates Pandas dataframe from results that is then saved to a .csv file.
    The resulting dataframe can be used with searchstims.plot.figures.acc_v_set_size_df

    Parameters
    ----------
    data_prep_dir : pathlib.Path
        used to find paths to .gz files containing results, if not supplied via data_gz_paths argument
    results_dir : pathlib.Path
        used to find paths to .gz files containing results, if not supplied via results_gz_paths argument
    test_csv_path : str
        saved csv will have this filename, can include complete path.
        Default is './test.csv'
    nets : list
        of string, names of neural network architectures.
        Default is ('alexnet', 'VGG16')
    train_types : list
        of string, training methods used. Expected to be in data and results filenames.
        Default is ('finetune', 'train')
    stims : list
        of str, names of visual search stimuli.
        Default is ('2_v_5', 'RVvGV', 'RVvRHGV').
    target_condition
    data_gz_paths : list
        of str, paths to data files. In case they have different names than what would be determined programatically.
    results_gz_paths
        of str, paths to results files. In case they have different names than what would be determined programatically.

    Returns
    -------
    None

    saves Pandas dataframe as .csv file using test_csv_path as filename
    """

    if results_gz_paths and data_gz_paths:
        if len(results_gz_paths) != len(data_gz_paths):
            raise ValueError(
                'results_gz_paths must be same length as data_gz_paths'
            )

    num_iter_main_loop = len(train_types) * len(nets) * len(stims)
    if len(results_gz_paths) != num_iter_main_loop:
        raise ValueError(
            f'not enough paths in results_gz_paths (length {len(results_gz_paths)} '
            f'for all iterations of main loop, {num_iter_main_loop}.'
        )

    iter_counter = 0
    rows = []
    for train_type in train_types:
        for net in nets:
            for stim in stims:
                if data_gz_paths:
                    data_gz_path = data_gz_paths[iter_counter]
                else:
                    data_gz_path = data_prep_dir.joinpath(f'{net}_{train_type}_{stim}_data.gz')

                data_dict = joblib.load(data_gz_path)
                y_test = data_dict['y_test']
                set_size_vec_test = data_dict['set_size_vec_test']
                set_sizes = data_dict['set_sizes_by_stim_type'][stim]

                if results_gz_paths:
                    results_gz_path = results_gz_paths[iter_counter]
                else:
                    results_gz_path = list(results_dir.joinpath(f'{net}_{train_type}_{stim}').glob(f'*test*.gz'))
                    assert len(results_gz_path) == 1, 'found more than one results_gz_path with glob'
                    results_gz_path = results_gz_path[0]

                results_dict = joblib.load(results_gz_path)
                ppm = results_dict['predictions_per_model_dict']
                num_nets = len(ppm.keys())
                for net_num in range(num_nets):
                    key = [key for key in ppm.keys() if f'net_number_{net_num}' in key][0]
                    y_pred = ppm[key]
                    for target_cond in target_condition:
                        for set_size in set_sizes:
                            if target_cond == 'present':
                                inds_this_cond = np.where(
                                    np.logical_and(y_test == 1, set_size_vec_test == set_size))
                            elif target_cond == 'absent':
                                inds_this_cond = np.where(
                                    np.logical_and(y_test == 0, set_size_vec_test == set_size))
                            acc = np.sum(y_pred[inds_this_cond] == y_test[inds_this_cond])
                            acc = acc / y_test[inds_this_cond].shape[0]
                            row = [net, train_type, net_num, stim, target_cond, set_size, acc]
                            rows.append(row)

                iter_counter += 1

    test_df = pd.DataFrame.from_records(rows, columns=HEADER)
    test_df.to_csv(test_csv_path)
