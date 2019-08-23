# !/usr/bin/env python
# coding: utf-8
import json
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).parent
DATA_ROOT = HERE.joinpath('../../data')

SPLITS = ['train', 'val', 'test']


def list2vec(a_list):
    """convert a list representing part of a sharded dataset into one big numpy array"""
    if type(a_list) == np.ndarray:
        return a_list

    elif type(a_list) == list:
        if all([type(item) == list for item in a_list]):
            a_list = [item for sublist in a_list for item in sublist]

        if all([type(item) == str for item in a_list]):
            a_list = np.asarray(a_list)

        if all([type(item) == np.ndarray for item in a_list]):
            a_list = np.concatenate(a_list)

        return a_list
    else:
        raise TypeError('expected list or numpy array')


ALSO_ADD = ['set_sizes_by_stim_type', 'shard_train', 'shard_size']


def split_dataset(data_gz_fname, json_fname, train_mask, stim_abbrev, new_data_gz_name):
    data_gz = joblib.load(data_gz_fname)
    with open(json_fname) as fp:
        stim_meta_dict = json.load(fp)

    stim_meta_list = []
    stim_meta_dict = stim_meta_dict[stim_abbrev]
    for set_size, stim_meta_this_set_size in stim_meta_dict.items():
        # *** only using present because we only care about splitting target present condition up ***
        stim_meta_list.extend(stim_meta_this_set_size['present'])

    fname_grid_map = {}
    for meta_d in stim_meta_list:
        stim_fname_meta = Path(meta_d['filename']).name
        char_grid = np.asarray(meta_d['grid_as_char'])
        fname_grid_map[stim_fname_meta] = char_grid

    # make new train set in a dictionary because it's easier to loop over keys later
    # instead of repeating ourselves with different variable names getting transformed the same way
    keys = ['x', 'y', 'set_size_vec']
    splits_new = {
        'train': {key: [] for key in keys},
    }

    x_train = list2vec(data_gz['x_train'])
    y_train = list2vec(data_gz['y_train'])
    set_size_vec_train = list2vec(data_gz['set_size_vec_train'])
    for fname, target_present, set_size in zip(x_train, y_train, set_size_vec_train):
        if 'present' in fname:
            fname_name = Path(fname).name
            char_grid = fname_grid_map[fname_name]
            if np.any(np.logical_and(char_grid == 't', train_mask)):
                splits_new['train']['x'].append(fname)
                splits_new['train']['set_size_vec'].append(set_size)
                splits_new['train']['y'].append(target_present)
        elif 'absent' in fname:
            splits_new['train']['x'].append(fname)
            splits_new['train']['set_size_vec'].append(set_size)
            splits_new['train']['y'].append(target_present)

    # make sure training set is balanced w/same # of target present/absent for each set size
    set_size_vec = np.asarray(splits_new['train']['set_size_vec'])
    set_sizes = np.unique(set_size_vec)
    y_train_new_arr = np.asarray(splits_new['train']['y'])
    keep_inds = []

    for set_size in set_sizes:
        inds_this_set_size_target_present = np.nonzero(
            np.logical_and(set_size_vec == set_size, y_train_new_arr == 1)
        )[0]
        inds_this_set_size_target_absent = np.nonzero(
            np.logical_and(set_size_vec == set_size, y_train_new_arr == 0)
        )[0]
        num_present = inds_this_set_size_target_present.shape[0]
        num_absent = inds_this_set_size_target_absent.shape[0]
        if num_present < num_absent:
            inds_this_set_size_target_absent = inds_this_set_size_target_absent[:num_present]
        elif num_present > num_absent:
            inds_this_set_size_target_present = inds_this_set_size_target_present[:num_absent]
        else:
            pass
        keep_inds.extend(inds_this_set_size_target_absent.tolist())
        keep_inds.extend(inds_this_set_size_target_present.tolist())

    keep_inds = np.asarray(keep_inds)
    for key in keys:
        as_arr = np.asarray(splits_new['train'][key])
        splits_new['train'][key] = as_arr[keep_inds].tolist()

    # split training set into shards, if necessary
    if data_gz['shard_train']:
        shard_size = data_gz['shard_size']
        # get floor to figure out num samples per shard for each set size,
        # and then we'll throw any leftovers into the last (num_shards + 1) shard
        set_sizes, set_size_samples_per_shard = np.unique(data_gz['set_size_vec_train'][0], return_counts=True)

        x_train = np.asarray(splits_new['train']['x'])
        y_train = np.asarray(splits_new['train']['y'])
        set_size_vec_train = np.asarray(splits_new['train']['set_size_vec'])
        for_sharding = {int(set_size): {} for set_size in
                        set_sizes}  # will add 'present' and 'absent' keys in next loop below
        for set_size in set_sizes:
            set_size_inds = np.nonzero(set_size_vec_train == set_size)[0]
            set_size_present_inds = np.nonzero(y_train[set_size_inds] == 1)[0]
            set_size_absent_inds = np.nonzero(y_train[set_size_inds] == 1)[0]
            for_sharding[int(set_size)]['present'] = x_train[set_size_present_inds].tolist()
            for_sharding[int(set_size)]['absent'] = x_train[set_size_absent_inds].tolist()

        for set_size, num_samples in zip(set_sizes, set_size_samples_per_shard):
            is_odd = num_samples % 2
            if is_odd:
                coin_flip = np.random.choice([0, 1])
                if coin_flip:
                    n_present = int(np.ceil(num_samples / 2))
                    n_absent = num_samples - n_present
                else:
                    n_absent = int(np.ceil(num_samples / 2))
                    n_present = num_samples - n_absent
            else:
                n_present = n_absent = int(num_samples / 2)
            set_size = int(set_size)
            total_present = len(for_sharding[set_size]['present'])
            for_sharding[set_size]['present'] = [
                for_sharding[set_size]['present'][i:i + n_present] for i in range(0, total_present, n_present)]
            total_absent = len(for_sharding[set_size]['absent'])
            for_sharding[set_size]['absent'] = [
                for_sharding[set_size]['absent'][i:i + n_absent] for i in range(0, total_absent, n_absent)]

        x_sharded = []
        y_sharded = []
        set_size_sharded = []
        num_shards_now = set(
            len(for_sharding[set_size][target_cond])
            for set_size in set_sizes
            for target_cond in ['present', 'absent']
        )
        if len(num_shards_now) != 1:
            # because somehow some set sizes ended up with more than others
            num_shards_now = np.min(list(num_shards_now))  # keep same number for all set sizes
        else:
            num_shards_now = num_shards_now.pop()

        for shard_ind in range(num_shards_now):
            x_shard = []
            y_shard = []
            set_size_shard = []
            for set_size in set_sizes:
                set_size = int(set_size)
                x_present = for_sharding[set_size]['present'][shard_ind]
                x_shard.extend(x_present)
                y_shard.extend([1 for el in x_present])
                set_size_shard.extend([set_size for el in x_present])

                x_absent = for_sharding[set_size]['absent'][shard_ind]
                x_shard.extend(x_absent)
                y_shard.extend([0 for el in x_absent])
                set_size_shard.extend([set_size for el in x_absent])

            y_shard = np.asarray(y_shard)
            set_size_shard = np.asarray(set_size_shard)
            x_sharded.append(x_shard)
            y_sharded.append(y_shard)
            set_size_sharded.append(set_size_shard)

        splits_new['train']['x'] = x_sharded
        splits_new['train']['y'] = y_sharded
        splits_new['train']['set_size_vec'] = set_size_sharded

        # also remake data_gz with same number of shards (i.e. with less shards)
        # so we can train with this and know any difference is not due to difference in number of training samples
        data_gz_less_shards = joblib.load(data_gz_fname)
        for key in keys:
            data_gz_less_shards[f'{key}_train'] = data_gz_less_shards[f'{key}_train'][:num_shards_now]
        data_gz_less_shards_fname = str(
            new_data_gz_name.parent.joinpath(data_gz_fname.name.replace('.gz', '_less_shards.gz'))
        )
        joblib.dump(data_gz_less_shards, data_gz_less_shards_fname)

    else:  # if shard_train is not True
        # keep x as a list but
        splits_new['train']['y'] = np.asarray(splits_new['train']['y'])
        splits_new['train']['set_size_vec'] = np.asarray(splits_new['train']['set_size_vec'])

    # finally make the new 'data dict' we will save
    out_dict = {}

    for split in SPLITS:
        if split == 'train':
            for key, a_list in splits_new['train'].items():
                out_dict[f'{key}_{split}'] = splits_new[split][key]
        else:
            for key in keys:
                # just arrays directly for validation and test sets
                out_dict[f'{key}_{split}'] = data_gz[f'{key}_{split}']

    for other_key in ALSO_ADD:
        out_dict[other_key] = data_gz[other_key]

    joblib.dump(out_dict, new_data_gz_name)


GRID_SHAPE = (5, 5)

TRAIN_MASK = np.zeros(GRID_SHAPE).astype(np.int32)
TRAIN_MASK[:, :3] = 1


def main():
    data_gz_fnames = [
        'alexnet_train_RVvGV_data.gz',
        'alexnet_train_RVvRHGV_data.gz',
        'alexnet_train_2_v_5_data.gz',
    ]
    json_fnames = [
        'alexnet_train_RVvGV/alexnet_train_RVvGV.json',
        'alexnet_train_RVvRHGV/alexnet_train_RVvRHGV.json',
        'alexnet_train_2_v_5/alexnet_train_2_v_5.json',
    ]
    stim_abbrevs = [
        'RVvGV',
        'RVvRHGV',
        '2_v_5',
    ]

    for data_gz_fname, json_fname, stim_abbrev in zip(data_gz_fnames, json_fnames, stim_abbrevs):
        # before we turn data_gz_fname into a path, let's make new data gz fname
        new_data_gz_name = data_gz_fname.replace('train', 'train_test_target_split')

        data_gz_fname = DATA_ROOT.joinpath(
            f'data_prepd_for_nets/{data_gz_fname}'
        )
        json_fname = DATA_ROOT.joinpath(
            f'visual_search_stimuli/{json_fname}'
        )

        new_data_gz_name = DATA_ROOT.joinpath(
            f'expt_13/data_prepd_for_nets/{new_data_gz_name}'
        )

        print(f'splitting dataset: {data_gz_fname}')
        split_dataset(data_gz_fname, json_fname, TRAIN_MASK, stim_abbrev, new_data_gz_name)


if __name__ == '__main__':
    main()
