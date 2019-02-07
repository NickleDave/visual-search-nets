import os
import json
from glob import glob

import numpy as np
import imageio
import joblib


def data(config):
    """prepare training, validation, and test datasets.
    Saves a compressed Python dictionary in the path specified
    by the NPZ_FILENAME option in the DATA section of a config.ini file.

    Parameters
    ----------
    config

    Returns
    -------
    None

    Notes
    -----
    The dictionary saved has the following key-value pairs:
        x_train : np.ndarray
            images used as input to train neural network
        y_train : np.ndarray
            labels, expected output of neural network.
            Either element is either 1, meaning "target present", or 0, "target absent".
        x_val : np.ndarray
            images for validation set used during training
        y_val : np.ndarray
            labels for validation set
        x_test : np.ndarray
            images for set used to test accuracy of trained network
        y_test : np.ndarray
            labels for set used to test accuracy of trained network
        train_set_files : np.ndarray
            filenames corresponding to images in x_train
        val_set_files : np.ndarray
            filenames corresponding to images in x_val
        test_set_files : np.ndarray
            filenames corresponding to images in x_test
        set_size_vec_train : np.ndarray
            "set size" of each image in x_train,
            total number of targets + distractors.
        set_size_vec_val : np.ndarray
            set size of each image in x_val
        set_size_vec_test : np.ndarray
            set size of each image in x_test
        set_sizes : list
            ordered set of unique set sizes.
            Useful if you need to plot accuracy v. set size.
    """
    train_dir = config['DATA']['TRAIN_DIR']
    stim_type = config['DATA']['STIM_TYPE']

    fname_json = glob(os.path.join(train_dir, '*.json'))

    if not fname_json:
        raise ValueError("couldn't find .json file with stimulus filenames in {}"
                         .format(train_dir))
    elif len(fname_json) > 1:
        raise ValueError("found more than one .json file with stimulus filenames in {}"
                         .format(train_dir))
    else:
        fname_json = fname_json[0]

    with open(fname_json) as f:
        stim_info_by_set_size = json.load(f)
    stim_info_by_set_size = {int(k): v for k, v in stim_info_by_set_size.items()}

    set_sizes = [k for k in stim_info_by_set_size.keys()]

    train_size = int(config['DATA']['TRAIN_SIZE'])
    train_size_per_set_size = (train_size / len(set_sizes)) / 2
    if train_size_per_set_size.is_integer():
        train_size_per_set_size = int(train_size_per_set_size)
    else:
        raise TypeError('train_size_per_set_size is not a whole number, adjust '
                        'total number of samples, or number of set sizes.')
    if config.has_option('DATA', 'VALIDATION_SIZE'):
        val_size = int(config['DATA']['VALIDATION_SIZE'])
        val_size_per_set_size = (val_size / len(set_sizes)) / 2
        if val_size_per_set_size.is_integer():
            val_size_per_set_size = int(val_size_per_set_size)
        else:
            raise TypeError('val_size_per_set_size is not a whole number, adjust '
                            'total number of samples, or number of set sizes.')
    else:
        val_size = Nonetrin

    train_set_files = []
    # initialize list to convert into a
    # vector that will indicates set size
    # for each image (number of items present)
    # i.e. np.array([1, 1, 1, ..., 4, 4, 4, ... 8, 8])
    set_size_vec_train = []
    test_set_files = []
    set_size_vec_test = []
    if val_size:
        val_set_files = []
        set_size_vec_val = []
    else:
        val_set_files = None
        set_size_vec_val = None

    # the dict comprehension below contains some hard-to-comprehend unpacking
    # of 'stim_info_by_set_size', so we can just keep the filenames.
    # The structure of the .json file is a dict of dicts (see the searchstims
    # docs for more info). The net effect of the unpacking is that each
    # `present_absent_dict` is a dict with 'present' and
    # 'absent' keys. Value for each key is a list of filenames of images
    # where target is either present (if key is 'present') or absent
    stim_fnames_by_set_size = {
        set_size: {present_or_absent: [
            stim_info_dict['filename'] for stim_info_dict in stim_info_list
        ]
            for present_or_absent, stim_info_list in present_absent_dict.items()

        }
        for set_size, present_absent_dict in stim_info_by_set_size.items()
    }

    # do some extra juggling to make sure we have equal number of target present
    # and target absent stimuli for each "set size", in training and test datasets
    for set_size, stim_fnames_present_absent in stim_fnames_by_set_size.items():
        for present_or_absent, stim_fnames in stim_fnames_present_absent.items():
            inds = np.arange(len(stim_fnames))
            if val_size:
                not_test_inds = np.random.choice(inds,
                                                 size=train_size_per_set_size +
                                                      val_size_per_set_size,
                                                 replace=False)
                train_inds_bool = np.isin(inds, not_test_inds[:train_size_per_set_size])
                val_inds_bool = np.isin(inds, not_test_inds[-val_size_per_set_size:])
            else:
                not_test_inds = np.random.choice(inds,
                                                 size=train_size_per_set_size,
                                                 replace=False)
                train_inds_bool = np.isin(inds, not_test_inds)
            test_inds_bool = np.isin(inds, not_test_inds, invert=True)

            stim_fnames_arr = np.asarray(stim_fnames)
            tmp_train_set_files = stim_fnames_arr[train_inds_bool].tolist()
            train_set_files.extend(tmp_train_set_files)
            set_size_vec_train.extend([set_size] * len(tmp_train_set_files))
            if val_size:
                tmp_val_set_files = stim_fnames_arr[val_inds_bool].tolist()
                val_set_files.extend(tmp_val_set_files)
                set_size_vec_val.extend([set_size] * len(tmp_val_set_files))
            tmp_test_set_files = stim_fnames_arr[test_inds_bool].tolist()
            test_set_files.extend(tmp_test_set_files)
            set_size_vec_test.extend([set_size] * len(tmp_test_set_files))

    set_size_vec_train = np.asarray(set_size_vec_train)
    if val_size:
        set_size_vec_val = np.asarray(set_size_vec_val)
    set_size_vec_test = np.asarray(set_size_vec_test)

    print('loading images for training set')
    x_train = []
    for fname in train_set_files:
        x_train.append(imageio.imread(fname))
    x_train = np.asarray(x_train)

    if val_size:
        print('loading images for validation set')
        x_val = []
        for fname in val_set_files:
            x_val.append(imageio.imread(fname))
        x_val = np.asarray(x_val)
    else:
        x_val = None

    print('loading images for test set')
    x_test = []
    for fname in test_set_files:
        x_test.append(imageio.imread(fname))
    x_test = np.asarray(x_test)

    y_train = np.asarray(['present' in fname for fname in train_set_files],
                         dtype=int)

    if val_size:
        y_val = np.asarray(['present' in fname for fname in val_set_files],
                           dtype=int)
    else:
        y_val = None

    y_test = np.asarray(['present' in fname for fname in test_set_files],
                        dtype=int)

    npz_filename = config['DATA']['NPZ_FILENAME']

    data_dict = dict(x_train=x_train,
                     y_train=y_train,
                     x_val=x_val,
                     y_val=y_val,
                     x_test=x_test,
                     y_test=y_test,
                     train_set_files=train_set_files,
                     val_set_files=val_set_files,
                     test_set_files=test_set_files,
                     set_size_vec_train=set_size_vec_train,
                     set_size_vec_val=set_size_vec_val,
                     set_size_vec_test=set_size_vec_test,
                     set_sizes=set_sizes,
                     )

    joblib.dump(npz_filename, data_dict)

