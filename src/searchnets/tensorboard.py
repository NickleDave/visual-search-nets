"""functions for working with tensorboard"""
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def logdir2df(logdir):
    """convert tensorboard events files in a logs directory into a pandas DataFrame

    events files are created by SummaryWriter from PyTorch or Tensorflow

    Parameters
    ----------
    logdir : str, Path
        path to directory containing tfevents file(s) saved by a SummaryWriter

    Returns
    -------
    df : pandas.Dataframe
        with columns 'step', 'wall_time', and all Scalars from the tfevents file

    Notes
    -----
    adapted from
    https://stackoverflow.com/questions/42355122/can-i-export-a-tensorflow-summary-to-csv
    """
    if issubclass(type(logdir), Path):  # subclass, because could be PosixPath or WindowsPath
        logdir = str(logdir)

    ea = EventAccumulator(path=logdir)
    ea.Reload()  # load all data written so far

    scalar_tags = ea.Tags()['scalars']  # list of tags for values written to scalar

    dfs = {}

    for scalar_tag in scalar_tags:
        dfs[scalar_tag] = pd.DataFrame(ea.Scalars(scalar_tag),
                                       columns=["wall_time",
                                                "step",
                                                scalar_tag.replace('val/', '')])
        dfs[scalar_tag] = dfs[scalar_tag].set_index("step")
        dfs[scalar_tag].drop("wall_time", axis=1, inplace=True)
    return pd.concat([v for k, v in dfs.items()], axis=1)


def logdir2csv(logdir):
    """convert tensorboard events files in a logs directory into a .csv file

    Parameters
    ----------
    logdir : str, Path
        path to directory containing tfevents file(s) saved by a SummaryWriter

    Returns
    -------
    None
    """
    logdir = Path(logdir)
    events_files = sorted(logdir.glob('*tfevents*'))
    # remove .csv files -- we can just overwrite them
    events_files = [path for path in events_files if not str(path).endswith('.csv')]
    if len(events_files) != 1:
        if len(events_files) < 1:
            raise ValueError(
                f'did not find any events files in {logdir}'
            )
        elif len(events_files) > 1:
            raise ValueError(
                f'found multiple events files in {logdir}:\n{events_files}.'
                'Please ensure there is only one events file in the directory, '
                'unclear which to use.'
            )
    else:
        events_file = events_files[0]

    df = logdir2df(logdir)

    csv_path = events_file.parent.joinpath(events_file.stem + '.csv')
    df.to_csv(logdir.joinpath(csv_path))
