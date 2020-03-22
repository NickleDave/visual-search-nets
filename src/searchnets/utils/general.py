import os
from pathlib import Path
import tarfile

from pyprojroot import here


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


def projroot_path(relative_path):
    """convert relative path to path with root set to project root

    just a wrapper around pyprojroot.here, used as converter in config and elsewhere

    Parameters
    ----------
    relative_path : str, Path

    Returns
    -------
    projroot_path : Path
        relative to project root, as determined by pyprojroot package
    """
    return here(relative_project_path=relative_path)
