# visual-search-nets

Experiments to measure the behavior of deep neural networks performing 
a visual search task.

For some background and a summary of the results, please see [this Jupyter notebook](./docs/notebooks/results.ipynb).

## Installation
Experiments were run using Anaconda on Ubuntu 16.04.
The following commands were used to create the environment:

```console
tu@computi:~$ conda create -n searchnets python=3.6 numpy matplotlib imageio joblib tensorflow-gpu 
tu@computi:~$ source activate searchnets
tu@computi:~$ git clone https://github.com/NickleDave/visual-search-nets.git
tu@computi:~$ cd ./visual-search-nets
tu@computi:~$ pip install .
```

## Data
Data is deposited here:
<https://figshare.com/articles/visual-search-nets/7688840>

## Replicating experiments
The `Makefile` replicates the experiments.
```console
tu@computi:~$ make all
```
