import os
import json
from glob import glob
from math import ceil
import random

import numpy as np
import joblib


def data(train_dir,
         train_size,
         gz_filename,
         stim_types=None,
         val_size=None,
         test_size=None,
         train_size_per_set_size=None,
         val_size_per_set_size=None,
         test_size_per_set_size=None,
         shard_train=False,
         shard_size=None):
    """prepare training, validation, and test datasets.
    Saves a compressed Python dictionary in the path specified
    by the GZ_FILENAME option in the DATA section of a config.ini file.

    Parameters
    ----------
    train_dir : str
        path to directory where training data is saved
    train_size : int
        number of samples to put in training data set.
        Note that if there is more than on stimulus type in the source data set, then the number of stimuli from
        each type will be train_size / number of types.
    gz_filename : str
        name of .gz file in which to save dataset
    stim_types : list
        of strings; specifies which visual search stimulus types to use when creating dataset. Strings must be keys in
        .json file within train_dir. Default is None, in which case all types found in .json file will be used.
    val_size : int
        number of samples to put in validation data set.
        Default is None, in which case there will be no validation set.
        Note that if there is more than on stimulus type in the source data set, then the number of stimuli from
        each type will be val_size / number of types.
    test_size : int
        number of samples to put in test data set.
        Default is None, in which case all samples not used in training and validation sets are used for test set.
        Note that if there is more than on stimulus type in the source data set, then the number of stimuli from
        each type will be test_size / number of types.

    Returns
    -------
    None

    Notes
    -----
    The dictionary saved has the following key-value pairs:
        x_train : np.ndarray
            filenames corresponding to images used as input to train neural network
        y_train : np.ndarray
            labels, expected output of neural network.
            Either element is either 1, meaning "target present", or 0, "target absent".
        x_val : np.ndarray
            filenames corresponding to images for validation set used during training
        y_val : np.ndarray
            labels for validation set
        x_test : np.ndarray
            filenames corresponding to images for set used to test accuracy of trained network
        y_test : np.ndarray
            labels for set used to test accuracy of trained network
        set_size_vec_train : np.ndarray
            "set size" of each image in x_train,
            total number of targets + distractors.
        set_size_vec_val : np.ndarray
            set size of each image in x_val
        set_size_vec_test : np.ndarray
            set size of each image in x_test
        set_sizes_by_stim : dict
            ordered set of unique set sizes, mapped to each stimulus type
            Useful if you need to plot accuracy v. set size.
        set_sizes_by_stim_stype : numpy.ndarray
        stim_type_vec_train : numpy.ndarray
            type of visual search stimulus for each image in x_train.
            Needed when there are multiple types of visual search stimuli in the dataset.
        stim_type_vec_val : numpy.ndarray
            type of visual search stimulus for each image in x_val.
        stim_type_vec_test : numpy.ndarray
            type of visual search stimulus for each image in x_test.
    """
    fname_json = glob(os.path.join(train_dir, '*.json'))

    if not fname_json:
        raise ValueError("couldn't find .json file with stimulus filenames in {}"
                         .format(train_dir))
    elif len(fname_json) > 1:
        raise ValueError("found more than one .json file with stimulus filenames in {}"
                         .format(train_dir))
    else:
        fname_json = fname_json[0]

    x_train = []
    # initialize list to convert into a
    # vector that will indicates set size
    # for each image (number of items present)
    # i.e. np.array([1, 1, 1, ..., 4, 4, 4, ... 8, 8])
    set_size_vec_train = []
    stim_type_vec_train = []
    if val_size:
        x_val = []
        set_size_vec_val = []
        stim_type_vec_val = []
    else:
        x_val = None
        set_size_vec_val = None
    x_test = []
    set_size_vec_test = []
    stim_type_vec_test = []

    with open(fname_json) as f:
        fnames_dict = json.load(f)

    if stim_types:
        if type(stim_types) != list or not all([type(stim_type) == str for stim_type in stim_types]):
            raise TypeError('stim_types must be a list of strings')
        if not all([stim_type in fnames_dict for stim_type in stim_types]):
            not_in_fnames_dict = [stim_type for stim_type in stim_types if stim_type not in fnames_dict]
            raise ValueError(
                f'the following stimulus types were not found in {fname_json}: {not_in_fnames_dict}'
            )
        # just keep the stimuli specified in stim_types
        fnames_dict = {k: v for k, v in fnames_dict.items() if k in stim_types}
    else:
        stim_types = list(fnames_dict.keys())  # number of keys in fnames_dict will be number of stim
    num_stim_types = len(stim_types)

    train_size_per_stim_type = train_size / num_stim_types
    if train_size_per_stim_type.is_integer():
        train_size_per_stim_type = int(train_size_per_stim_type)
    else:
        raise TypeError(f'train_size_per_stim_type, {train_size_per_stim_type}, is is not a whole number.\n'
                        'It is calculated as: (train_size / number of visual search stimulus types))\n'
                        'Adjust total number of samples, or number of stimulus types.')

    if train_size_per_set_size:
        total_train_size_from_per_set_size = sum(train_size_per_set_size)
        if total_train_size_from_per_set_size != train_size_per_stim_type:
            raise ValueError(
                f'total number of training samples specified in '
                f'train_size_per_set_size, {total_train_size_from_per_set_size} does not equal number determined '
                f'by dividing train_size up by number of stim_types: {train_size_per_stim_type}'
            )

    if val_size:
        val_size_per_stim_type = val_size / num_stim_types
        if val_size_per_stim_type.is_integer():
            val_size_per_stim_type = int(val_size_per_stim_type)
        else:
            raise TypeError('val_size_per_set_size is not a whole number, adjust '
                            'total number of samples, or number of set sizes.')
    else:
        val_size_per_stim_type = 0

    if test_size:
        test_size_per_stim_type = test_size / num_stim_types
        if test_size_per_stim_type.is_integer():
            test_size_per_stim_type=int(test_size_per_stim_type)
        else:
            raise TypeError('test_size_per_set_size is not a whole number, adjust '
                            'total number of samples, or number of set sizes.')
    else:
        # "-1" means "use the remaining samples for the test set"
        test_size_per_stim_type = -1

    set_sizes_by_stim_type = {}

    for stim_type, stim_info_by_set_size in fnames_dict.items():
        # and this will be set sizes declared by user for this stimulus (could be diff't for each stimulus type).
        # First have to convert set size from char to int
        stim_info_by_set_size = {int(k): v for k, v in stim_info_by_set_size.items()}

        set_sizes = [k for k in stim_info_by_set_size.keys()]
        set_sizes_by_stim_type[stim_type] = set_sizes

        if train_size_per_set_size is None:
            train_size_per_set_size_this_stim = (train_size_per_stim_type / len(set_sizes)) / 2
            if train_size_per_set_size_this_stim.is_integer():
                train_size_per_set_size_this_stim = int(train_size_per_set_size_this_stim)
                train_size_per_set_size_this_stim = [train_size_per_set_size_this_stim for _ in set_sizes]
            else:
                raise TypeError(f'train_size_per_set_size, {train_size_per_set_size}, is is not a whole number.\n'
                                'It is calculated as: (train_size_per_stim_type / len(set_sizes)) / 2\n'
                                '(2 is for target present or absent).\n'
                                'Adjust total number of samples, or number of set sizes.')
        else:
            # if train_size_per_set_size is not None, divide each element in two (for target present / absent)
            train_size_per_set_size_this_stim = [item // 2 if item % 2 == 0 else item / 2
                                                 for item in train_size_per_set_size]

        if val_size:
            if val_size_per_set_size is None:
                val_size_per_set_size_this_stim = (val_size_per_stim_type / len(set_sizes)) / 2
                if val_size_per_set_size_this_stim.is_integer():
                    val_size_per_set_size_this_stim = int(val_size_per_set_size_this_stim)
                    val_size_per_set_size_this_stim = [val_size_per_set_size_this_stim for _ in set_sizes]
                else:
                    raise TypeError(f'val_size_per_set_size, {val_size_per_set_size},is not a whole number, adjust '
                                    'total number of samples, or number of set sizes.')
            else:
                # if val_size_per_set_size is not None, divide each element in two (for target present / absent)
                val_size_per_set_size_this_stim = [item // 2 if item % 2 == 0 else item / 2
                                                   for item in val_size_per_set_size]
        else:
            val_size_per_set_size_this_stim = [0 for _ in set_sizes]

        if test_size is None or test_size == -1:
            # "-1" means "use the remaining samples for the test set"
            test_size_per_set_size_this_stim = [-1 for _ in set_sizes]
        elif test_size > 0:
            if test_size_per_set_size is None:
                test_size_per_set_size_this_stim = (test_size_per_stim_type / len(set_sizes)) / 2
                if test_size_per_set_size_this_stim.is_integer():
                    test_size_per_set_size_this_stim = int(test_size_per_set_size_this_stim)
                    test_size_per_set_size_this_stim = [test_size_per_set_size_this_stim for _ in set_sizes]
                else:
                    raise TypeError(f'test_size_per_set_size, {test_size_per_set_size},is not a whole number, adjust '
                                    'total number of samples, or number of set sizes.')
            else:
                # if test_size_per_set_size is not None, divide each element in two (for target present / absent)
                test_size_per_set_size_this_stim = [item // 2 if item % 2 == 0 else item / 2
                                                    for item in test_size_per_set_size]
        else:
            raise ValueError(f'invalid test size: {test_size}')

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
        for ((set_size, stim_fnames_present_absent),
             train_size,
             val_size,
             test_size) in zip(stim_fnames_by_set_size.items(),
                               train_size_per_set_size_this_stim,
                               val_size_per_set_size_this_stim,
                               test_size_per_set_size_this_stim):
            for (present_or_absent, stim_fnames) in stim_fnames_present_absent.items():
                total_size = sum([size
                                  for size in (train_size, val_size, test_size)
                                  if size is not 0 and size is not -1])
                if total_size > len(stim_fnames):
                    raise ValueError(
                        f'number of samples for training + validation set, {total_size}, '
                        f'is larger than number of samples in data set, {len(stim_fnames)}'
                    )
                stim_fnames_arr = np.asarray(stim_fnames)

                inds = list(range(len(stim_fnames)))
                random.shuffle(inds)
                train_inds = np.asarray(
                    [inds.pop() for _ in range(train_size)]
                )
                tmp_x_train = stim_fnames_arr[train_inds].tolist()
                x_train.extend(tmp_x_train)
                set_size_vec_train.extend([set_size] * len(tmp_x_train))
                stim_type_vec_train.extend([stim_type] * len(tmp_x_train))

                if val_size > 0:
                    val_inds = np.asarray(
                        [inds.pop() for _ in range(val_size)]
                    )
                    tmp_x_val = stim_fnames_arr[val_inds].tolist()
                    x_val.extend(tmp_x_val)
                    set_size_vec_val.extend([set_size] * len(tmp_x_val))
                    stim_type_vec_val.extend([stim_type] * len(tmp_x_val))

                if test_size > 0:
                    test_inds = np.asarray([inds.pop() for _ in range(test_size)])
                elif test_size == -1:
                    # "-1" means "use the remaining samples for the test set"
                    test_inds = np.asarray([ind for ind in inds])

                tmp_x_test = stim_fnames_arr[test_inds].tolist()
                x_test.extend(tmp_x_test)
                set_size_vec_test.extend([set_size] * len(tmp_x_test))
                stim_type_vec_train.extend([stim_type] * len(tmp_x_test))

    x_train = [os.path.join(os.path.abspath(train_dir), path) for path in x_train]
    if x_val is not None:
        x_val = [os.path.join(os.path.abspath(train_dir), path) for path in x_val]
    x_test = [os.path.join(os.path.abspath(train_dir), path) for path in x_test]

    set_size_vec_train = np.asarray(set_size_vec_train)
    stim_type_vec_train = np.asarray(stim_type_vec_train)
    if val_size:
        set_size_vec_val = np.asarray(set_size_vec_val)
        stim_type_vec_val = np.asarray(stim_type_vec_val)
    set_size_vec_test = np.asarray(set_size_vec_test)
    stim_type_vec_test = np.asarray(stim_type_vec_test)

    y_train = np.asarray(['present' in fname for fname in x_train],
                         dtype=int)

    if shard_train:
        inds = np.arange(len(x_train))
        np.random.shuffle(inds)
        num_shards = int(ceil(len(x_train) / shard_size))
        x_train = np.asarray(x_train)  # so we can index with index arrays
        x_train_shards = []
        y_train_shards = []
        set_size_vec_train_shards = []
        stim_type_vec_train_shards = []
        for shard_num in range(num_shards):
            start = shard_size * shard_num
            stop = shard_size * (shard_num + 1)
            inds_this_shard = inds[start:stop]
            x_train_shards.append(x_train[inds_this_shard])
            y_train_shards.append(y_train[inds_this_shard])
            set_size_vec_train_shards.append(set_size_vec_train[inds_this_shard])
            stim_type_vec_train_shards.append(stim_type_vec_train[inds_this_shard])

        x_train = [x_train_shard.tolist() for x_train_shard in x_train_shards]
        y_train = y_train_shards
        set_size_vec_train = set_size_vec_train_shards
        stim_type_vec_train = stim_type_vec_train_shards

    if val_size:
        y_val = np.asarray(['present' in fname for fname in x_val],
                           dtype=int)
    else:
        y_val = None

    y_test = np.asarray(['present' in fname for fname in x_test],
                        dtype=int)

    gz_dirname = os.path.dirname(gz_filename)
    if not os.path.isdir(gz_dirname):
        os.makedirs(gz_dirname)

    data_dict = dict(x_train=x_train,
                     y_train=y_train,
                     x_val=x_val,
                     y_val=y_val,
                     x_test=x_test,
                     y_test=y_test,
                     set_size_vec_train=set_size_vec_train,
                     set_size_vec_val=set_size_vec_val,
                     set_size_vec_test=set_size_vec_test,
                     set_sizes_by_stim_type=set_sizes_by_stim_type,
                     stim_type_vec_train=stim_type_vec_train,
                     stim_type_vec_val=stim_type_vec_val,
                     stim_type_vec_test=stim_type_vec_test,
                     shard_train=shard_train,
                     shard_size=shard_size
                     )
    joblib.dump(data_dict, gz_filename)
