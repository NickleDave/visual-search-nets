import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def learning_curve(csv_fname, figsize=(15, 10), ylim=(0, 0.5), suptitle=None,
                   save_as=None):
    """plot results of learning curve

    Parameters
    ----------
    csv_fname
    figsize
    ylim
    suptitle
    save_as

    Returns
    -------
    None
    """
    df = pd.read_csv(csv_fname)
    set_sizes = sorted(df.set_size.unique())

    fig, ax = plt.subplots(1, len(set_sizes), figsize=figsize)
    for ax_ind, set_size in enumerate(set_sizes):
        df_this_ax = df.loc[df['set_size'] == set_size]
        sns.scatterplot(x='train_size', y='err', hue='setname', data=df_this_ax, legend=False, ax=ax[ax_ind])
        sns.lineplot(x='train_size', y='err', hue='setname', data=df_this_ax, ax=ax[ax_ind])
        ax[ax_ind].set_ylim(ylim)
        ax[ax_ind].set_xticks(df.train_size.unique())
        ax[ax_ind].set_xticklabels(df.train_size.unique(), rotation=45)
        ax[ax_ind].set_title(f'set size: {set_size}')
        if ax_ind == 0:
            ax[ax_ind].set_ylabel('error')
        else:
            ax[ax_ind].set_ylabel('')
        ax[ax_ind].set_xlabel('num. training samples')

    if suptitle:
        fig.suptitle(suptitle)

    if save_as:
        plt.savefig(save_as)
