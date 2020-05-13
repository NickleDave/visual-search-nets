from collections import defaultdict
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn.metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..datasets import VOCDetection
from ..transforms.util import get_transforms

from ..train import VSD_PAD_SIZE

HERE = Path(__file__).parent
DATA_ROOT = HERE.joinpath('../../../data')

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

    transform, target_transform = get_transforms(dataset_type='VSD', loss_func='BCE')

    # need to make Dataset so we know what ground truth labels are
    testset = VOCDetection(root=root,
                           csv_file=vsd_split_csv,
                           image_set='trainval',
                           split='test',
                           download=True,
                           transform=transform,
                           target_transform=target_transform,
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
