import matplotlib as mpl
import matplotlib.pyplot as plt

# set style etc. *before* we import plot functions
mpl.style.use('seaborn-dark')

plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.labelweight'] = 'regular'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['axes.axisbelow'] = True


from .heatmaps import item_heatmap
from .metric_v_set_size import acc_v_set_size, ftr_v_spt_conj,\
    metric_v_set_size_df, mn_slope_by_epoch
from .trainhistory import train_history
