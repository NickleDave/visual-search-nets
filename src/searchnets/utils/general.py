import csv
import tarfile
import os

import pandas
from statsmodels.formula.api import ols


def make_csv(eff_accs_10_epochs, ineff_accs_10_epochs, eff_accs_400_epochs, ineff_accs_400_epochs,
             set_sizes=(1, 2, 4, 8), csv_fname='results.csv'):
    """make csv from results

    Parameters
    ----------
    eff_accs_10_epochs : numpy.ndarray
    ineff_accs_10_epochs : numpy.ndarray
    eff_accs_400_epochs : numpy.ndarray
    ineff_accs_400_epochs : numpy.ndarray
    set_sizes : list
        of set sizes, in same order they appear in columns of array.
        Default is [1, 2, 4, 8].
    csv_fname : str
        name of .csv file to save

    Returns
    -------
    None
    """
    header = ["", "stim_type", "epochs", "replicate", "set_size", "acc"]
    out = [header]

    id = 1

    mats = [eff_accs_10_epochs, ineff_accs_10_epochs, eff_accs_400_epochs, ineff_accs_400_epochs]
    epochs = [10, 10, 400, 400]
    stim_type = ['eff', 'ineff', 'eff', 'ineff']
    for mat, epoch, stim in zip(mats, epochs, stim_type):
        for replicate, mat_row in enumerate(mat):
            for acc, set_size in zip(mat_row, set_sizes):
                csv_row = [id, stim, epoch, replicate, set_size, acc]
                out.append(csv_row)
                id += 1

    with open(csv_fname, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(out)


def anova(csv_fname):
    """run anova on data in csv file generated with make_csv"""
    data = pandas.read_csv(csv_fname)
    model = ols('acc ~ C(stim_type) + epochs + set_size', data).fit()
    print(model.summary())
    return model


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
