import matplotlib as mpl
import matplotlib.pyplot as plt

SET_SIZES = [1, 2, 4, 8]


def plot_results(eff_accs, ineff_accs, epochs, savefig=False, savedir=None):
    mpl.style.use('bmh')

    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.labelweight'] = 'regular'
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['figure.titlesize'] = 20

    fig, ax = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(10, 5)
    ax = ax.ravel()
    ax[0].plot(SET_SIZES, eff_accs.T)
    ax[0].set_xticks(SET_SIZES)
    ax[0].set_title('efficient')
    ax[0].set_xlabel('set size')
    ax[0].set_ylabel('accuracy')

    ax[1].plot(SET_SIZES, ineff_accs.T)
    ax[1].set_xticks(SET_SIZES)
    ax[1].set_title('inefficient')
    ax[1].set_xlabel('set size')
    ax[1].set_ylim([0, 1.1])

    plt.tight_layout()

    if savefig:
        fname = os.path.join(savedir, f'alexnet_eff_v_ineff_{epochs}_epochs.png')
        plt.savefig(fname)
