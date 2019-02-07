import csv

def make_csv(eff_accs_10_epochs, ineff_accs_10_epochs, eff_accs_400_epochs, ineff_accs_400_epochs,
             set_sizes=[1, 2, 4, 8], csv_fname='results.csv'):
    """make csv from results

    Parameters
    ----------
    eff_accs_10_epochs : np.ndarray
    ineff_accs_10_epochs : np.ndarray
    eff_accs_400_epochs : np.ndarray
    ineff_accs_400_epochs : np.ndarray
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
