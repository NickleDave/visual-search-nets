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


def results_csv(results_dir,
                data_prep_dir,
                test_csv_path='./test.csv',
                nets = ('alexnet', 'VGG16'),
                train_types = ('finetune', 'train'),
                stims = ('2_v_5', 'RVvGV', 'RVvRHGV'),
                target_condition = ('present', 'absent')):
    """make csv from results directory

    creates Pandas dataframe from results that is then saved to a .csv file.
    The resulting dataframe can be uesd with searchstims.plot.figures.acc_v_set_size_df
    """
    rows = []
    for train_type in train_types:
        for net in nets:
            for stim in stims:
                gz_fname = data_prep_dir.joinpath(f'{net}_{train_type}_{stim}_data.gz')
                data_dict = joblib.load(gz_fname)
                y_test = data_dict['y_test']
                set_size_vec_test = data_dict['set_size_vec_test']
                set_sizes = data_dict['set_sizes_by_stim_type'][stim]
                results_path = list(results_dir.joinpath(f'{net}_{train_type}_{stim}').glob(f'test_{net}*.gz'))
                assert len(results_path) == 1
                results_dict = joblib.load(results_path[0])
                for net_num in range(4):
                    ppm = results_dict['predictions_per_model_dict']
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
                            acc = np.sum(y_pred[inds_this_cond] == y_test[inds_this_cond]) / \
                                  y_test[inds_this_cond].shape[0]
                            row = [net, train_type, net_num, stim, target_cond, set_size, acc]
                            rows.append(row)

    test_df = pd.DataFrame.from_records(rows, columns=HEADER)
    test_df.to_csv(test_csv_path)
