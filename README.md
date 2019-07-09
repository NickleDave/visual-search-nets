[![DOI](https://zenodo.org/badge/169021695.svg)](https://zenodo.org/badge/latestdoi/169021695)
[![PyPI version](https://badge.fury.io/py/visual-search-nets.svg)](https://badge.fury.io/py/visual-search-nets)
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

## usage
Installing this package (by running `pip install .` in the source directory) makes it 
possible to run the experiments from the command line with the `searchnets` command, like so:
```console
tu@computi:~$ searchnets train config.ini
```  
The command-line interface accepts arguments with the syntax `searchnets command config.ini`,  
where `command` is some command to run, and `config.ini` is the name of a configuration file 
with options that specify how the command will be executed.  
For details on the commands, see [this page in the docs](./docs/cli.md).
For details on the `config.ini` files, please see [this other page](./docs/config.ini.md).

## Data
Data is deposited here:
<https://figshare.com/articles/visual-search-nets/7688840>

## Replicating experiments
The `Makefile` replicates the experiments.
```console
tu@computi:~$ make all
```

## Acknowledgements
- Research funded by the Lifelong Learning Machines program, 
DARPA/Microsystems Technology Office, 
DARPA cooperative agreement HR0011-18-2-0019
- David Nicholson was partially supported by the 
2017 William K. and Katherine W. Estes Fund to F. Pestilli, 
R. Goldstone and L. Smith, Indiana University Bloomington.

## Citation
Please cite the DOI for this code:
[![DOI](https://zenodo.org/badge/169021695.svg)](https://zenodo.org/badge/latestdoi/169021695)
