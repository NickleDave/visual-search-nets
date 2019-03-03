import tarfile
import os


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
