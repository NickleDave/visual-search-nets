from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn.metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from searchnets.datasets import VOCDetection
from searchnets.utils.transforms import VOCTransform

DATA_ROOT = Path('../../data')
vsd_results_dir = DATA_ROOT.joinpath('results/VSD_CORnet_Z_transfer_lr_1e-03_no_finetune')
vsd_cornet_z_results_gz = vsd_results_dir.joinpath('VSD_CORnet_Z_transfer_lr_1e-03_no_finetune_trained_200_epochs_test_results.gz')


def main():
    vsd_split_csv = DATA_ROOT.joinpath('Visual_Search_Difficulty_v1.0/VSD_dataset_split.csv')
    vsd_df = pd.read_csv(vsd_split_csv)

    vsd_df = vsd_df.drop('Unnamed: 0', axis=1)
    vsd_df_test = vsd_df[vsd_df['split'] == 'test']
    
    results = joblib.load(vsd_cornet_z_results_gz)
    # model keys are paths to checkpoint with saved model
    model_keys = list(results['img_names_per_model_dict'].keys())

    root = Path('~/Documents/data/voc')
    root = root.expanduser()
    csv_file = Path('~/Documents/repos/L2M/visual-search-nets/data/Visual_Search_Difficulty_v1.0/VSD_dataset_split.csv')
    csv_file = csv_file.expanduser()
    pad_size = 500

    # need to make Dataset so we know what ground truth labels are
    testset = VOCDetection(root=root,
                           csv_file=csv_file,
                           image_set='trainval',
                           split='test',
                           download=True,
                           transforms=VOCTransform(pad_size=pad_size),
                           return_img_name=True
                           )

    batch_size = 64
    num_workers = 32

    test_loader = DataLoader(testset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    # make sure that img names list will be the same for all models
    for model_key in model_keys:
        assert vsd_df_test['img'].values.tolist() == results['img_names_per_model_dict'][model_key]

    # grab one of them to use to find index for the img from each sample from the Dataset
    test_img_names = results['img_names_per_model_dict'][model_keys[0]]

    # also if img names list is the same as that for the dataframe (like we just asserted)
    # then we can use the same ind when we index into the new column we're making
    # for the dataframe, of f1 scores
    f1_scores = np.zeros(len(vsd_df_test))

    pbar = tqdm(test_loader)
    for i, sample in enumerate(pbar):
        # don't care about batch_x, just what y should be, and the img name
        _, batch_y, batch_img_name = sample
        # and we iterate through each sample in the batch
        pbar2 = tqdm(zip(batch_y, batch_img_name))
        for y, img_name in pbar2:
            y = y.cpu().numpy()  # convert to numpy array to pass to sklearn.metrics.f1_score
            y_pred = []
            y_true = []
            ind = test_img_names.index(img_name)  # use the image name to get its index from the list
            # and get predictions for that image ev **all** models!
            # (because we need votes from multiple models for an f1 score)
            for model_key in model_keys:
                y_pred.append(results['predictions_per_model_dict'][model_key][ind])
                y_true.append(y)
            y_pred = np.stack(y_pred)
            y_true = np.stack(y_true)
            f1 = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
            f1_scores[ind] = f1
            pbar2.set_description(f'f1 score: {f1}')

    vsd_df_test['f1'] = f1_scores
    vsd_df_test.to_csv('../../data/csv/VSD_CORnet_Z_transfer_lr_1e-03_no_finetune_test.csv')


if __name__ == '__main__':
    main()
