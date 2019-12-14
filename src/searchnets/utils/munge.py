import joblib
import numpy as np
import pandas as pd

from .metrics import compute_d_prime

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
