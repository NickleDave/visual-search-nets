from collections import defaultdict
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn.metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from searchnets.datasets import VOCDetection
from searchnets.utils.transforms import VOCTransform

from .metrics import compute_d_prime
from ..train import VSD_PAD_SIZE

HERE = Path(__file__).parent
DATA_ROOT = HERE.joinpath('../../../data')

# used by searchnets_results_df
SEARCHNETS_DF_COLUMNS = ['net_name',
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


def searchnets_results_df(data_csv_path,
                          results_gz_path,
                          net_name,
                          method,
                          learning_rate,
                          results_csv_path=None,
                          ):
    """creates Pandas dataframe from results of training models with searchstims dataset.
    The resulting dataframe can be used with searchnets.plot.figures.metric_v_set_size_df

    Parameters
    ----------
    data_csv_path : str
        path to .csv file of dataset splits, created by running 'searchnets split' from the command line.
    results_gz_path
        paths to .gz file created by running 'searchnets test' from the command line.
    net_name : str
        name of neural net architecture to train.
        One of {'alexnet', 'VGG16', 'CorNet Z'}
    method : str
        training method. One of {'initialize', 'transfer'}. See docstring for searchnets.train.
    learning_rate : float
        hyperparameter used during weight update step of training network.
    results_csv_path : str
        Path to use to save dataframe as a csv.
        Default is None, in which case no csv is saved.

    Returns
    -------
    df : pandas.DataFrame
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

    df = pd.DataFrame.from_records(rows, columns=SEARCHNETS_DF_COLUMNS)
    if results_csv_path:
        df.to_csv(results_csv_path)
    return df


DEFAULT_VOC_ROOT = Path('~/Documents/data/voc').expanduser()


def vsd_results_df(results_gz,
                   root=DEFAULT_VOC_ROOT,
                   pad_size=VSD_PAD_SIZE,
                   batch_size=64,
                   num_workers=32,
                   results_csv_path=None,
                   ):
    """creates Pandas dataframe from results of training models with Visual Search Difficulty dataset.
    The resulting dataframe can be used with searchnets.plot.figures.f1_v_vds_score_df

    Parameters
    ----------
    results_gz : str
        paths to .gz file created by running 'searchnets test' from the command line.
    root : str
        path to root of VOC dataset, as defined for torchvision.VOCDetection.
        Default is '~/Documents/data/voc'
    pad_size : int
        size of padding for images in Visual Search Difficulty dataset,
        applied by transform passed to Dataset in searchnets.train.
        Default is 500, declared as a constant in searchnets.train.
    batch_size : int
        Default is 64.
    num_workers : int
        Default is 32.
    results_csv_path : str
        Path to use to save dataframe as a csv.
        Default is None, in which case no csv is saved.

    Returns
    -------
    df : pandas.DataFrame

    Notes
    -----
    for each image in the 'test' subset of the Visual Search Difficulty dataset, this function computes the
    F1 score between the classes present in that image and the (multi-label) predictions of each trained network
    that are in the results_gz file. In addition it computes the arithmetic mean of F1 scores
    across all models. The individual F1 scores + mean F1 score are added as columns to the returned dataframe.
    """
    vsd_split_csv = DATA_ROOT.joinpath('Visual_Search_Difficulty_v1.0/VSD_dataset_split.csv')
    vsd_df = pd.read_csv(vsd_split_csv)
    vsd_df = vsd_df.drop('Unnamed: 0', axis=1)
    vsd_df_test = vsd_df[vsd_df['split'] == 'test']

    results = joblib.load(results_gz)
    # model paths to checkpoint with saved model, *** used as keys for dicts in results_gz files ***
    # in theory they should be sorted numerically already because they were added to the dictionary in the loop
    # and dict keys are insertion-ordered as of 3.6
    # but let's be extra paranoid and sort anyway!
    model_keys_for_results_gz = sorted(
        results['img_names_per_model_dict'].keys(),
        key=lambda x: int(x.name.split('_')[-1])
    )
    model_key_num_map = {}
    for model_key in model_keys_for_results_gz:
        model_key_num_map[model_key] = f'model_{int(model_key.name.split("_")[-1])}'

    # make sure that img names list will be the same for all models
    for model_key in model_keys_for_results_gz:
        assert vsd_df_test['img'].values.tolist() == results['img_names_per_model_dict'][model_key]

    # grab one of them to use to find index for the img from each sample from the Dataset
    test_img_names = results['img_names_per_model_dict'][model_keys_for_results_gz[0]]

    # need to make Dataset so we know what ground truth labels are
    testset = VOCDetection(root=root,
                           csv_file=vsd_split_csv,
                           image_set='trainval',
                           split='test',
                           download=True,
                           transforms=VOCTransform(pad_size=pad_size),
                           return_img_name=True
                           )

    test_loader = DataLoader(testset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    # also if img names list is the same as that for the dataframe (like we just asserted)
    # then we can use the same ind when we index into the new columns we're making
    # for the dataframe
    new_columns = defaultdict(
        # for any new key, default to an array the same length as our dataframe, will be a new column
        partial(np.zeros, shape=len(vsd_df_test))
    )

    pbar = tqdm(test_loader)
    n_batch = len(test_loader)
    for i, sample in enumerate(pbar):
        pbar.set_description(f'batch {i} of {n_batch}')
        # don't care about batch_x, just what y should be, and the img name
        _, batch_y, batch_img_name = sample
        # and we iterate through each sample in the batch
        for y, img_name in zip(batch_y, batch_img_name):
            y_true = y.cpu().numpy()  # convert to numpy array to pass to sklearn.metrics.f1_score
            row = test_img_names.index(img_name)  # use the image name to get its index from the list
            # and get predictions for that image from **all** models!
            # (because we need votes from multiple models for an f1 score)
            f1_scores_all_models = []
            acc_scores_all_models = []
            hamming_loss_all_models = []
            for model_key, model_num in model_key_num_map.items():
                y_pred = results['predictions_per_model_dict'][model_key][row]

                f1 = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
                new_columns[f'f1_score_{model_num}'][row] = f1
                f1_scores_all_models.append(f1)  # we'll use to get means after

                acc = sklearn.metrics.accuracy_score(y_true, y_pred)
                new_columns[f'acc_{model_num}'][row] = acc
                acc_scores_all_models.append(acc)

                hl = sklearn.metrics.hamming_loss(y_true, y_pred)
                new_columns[f'hamming_loss_{model_num}'][row] = hl
                hamming_loss_all_models.append(hl)

            mean_f1 = np.mean(f1_scores_all_models)
            new_columns['mean_f1_score'][row] = mean_f1
            mean_acc = np.mean(acc_scores_all_models)
            new_columns['mean_acc'][row] = mean_acc
            mean_hamming_loss = np.mean(hamming_loss_all_models)
            new_columns['mean_hamming_loss'][row] = mean_hamming_loss

    for column_name, values in new_columns.items():
        vsd_df_test[column_name] = values
    if results_csv_path:
        vsd_df_test.to_csv(results_csv_path)

    return vsd_df_test
