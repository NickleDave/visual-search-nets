# command-line interface

Installing this package (by running `pip install .` in the source directory) makes it 
possible to run the experiments from the command line with the `searchnets` command, like so:
```console
$ searchnets train config.ini
```  

The first argument to `searchnets` is a command, and the second is a `config.ini` file.
A single `config.ini` file corresponds to a single "experiment", so it will be used more than once, 
each time with a different command, as described in the sections below.
For details of the `.ini` files, see [this page](./config.ini.md)

## commands
Typically commands are run in the order they are presented here.

### `data`
Generates training, validation, and test data sets, and save them in a `.gz` file to be used 
by the `train` and `test` commands.

The files used to create the data sets are produced by first running the `searchstims` package, using 
the `config.ini` files supplied for that package in this repository.

### `train`
Train neural net architecture, as specified in `config.ini` file.
Trained checkpoints will be saved in subdirectories created in the directory 
specified by the `MODEL_SAVE_PATH`. There will be one sub-directory created for each number of 
epochs specified by the `EPOCHS` option, e.g. if `EPOCHS = 10` there will be a 
`trained_10_epochs` directory,  and if `EPOCHS = [10, 40]`, there would be 
an additional `trained_40_epochs` directory. 
Within each `trained_x_epochs` directory there will be one sub-directory for each "replicate" 
that contains the actual checkpoint files used to save and load train models. 

### `test`
Test accuracy of trained neural net architecture, as specified in `config.ini` file

### `all`
Run all three commands in order: `data`, then `train`, then `test`.
