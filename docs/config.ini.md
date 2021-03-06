# documentation

## config files
Experiments can be run with `config.ini` files. Some examples are in [./src/config](../src/config) file.
These `.ini` files have the following sections and options:

### `[DATA]` section
The `[DATA]` section provides options for preparing the data that is used to 
train the convolutional neural networks. The actual images must already have been generated using 
the `searchstims` package (<https://github.com/NickleDave/searchstims>). After generating those 
images, the user then uses this library to prepare data by executing the command `$ searchstims data name_of_config.ini` 
at the command line. This combines the images into Numpy arrays that are then saved in compressed format with 
the `joblib` library. The `[DATA]` section in the config.ini file must specify values for the following options:

* `TRAIN_DIR` : string  
  Path to directory where images generated by `searchstims` are saved.
* `TRAIN_SIZE` : integer  
  Number of images to include in training set.
* `VALIDATION_SIZE` : integer  
  Number of images to include in validation set. Used after each epoch to estimate accuracy of trained network.
* `SET_SIZES` : list  
  Specifies the 'set size' of the visual search stimuli.  The set size is the total number of targets and 
  distractors, AKA "items", in a visual search stimulus. Set sizes will already have been specified when 
  creating the images with `searchstims`, but here the user can choose to use all or only some of the available
  set sizes. Images for each set size are saved in separate folders by `searchstims`, so this list will be used 
  to find the paths to those folders within `TRAIN_DIR`.
* `GZ_FILENAME` : string  
  Path, including filename, where prepared data file should be saved. File is compressed with gzip using the 
  `joblib` library--to ensure this works correctly, this string should end with the right extension, `.gz` 

Here is an example `[DATA]` section:
```ini
[DATA]
TRAIN_DIR = ../data/visual_search_stimuli/searchstims_efficient
TRAIN_SIZE = 6200
VALIDATION_SIZE = 200
SET_SIZES = [1, 2, 4, 8]
GZ_FILENAME = ../data/data_prepd_for_nets/config_efficient_data.gz
```

### `[TRAIN]` section
The `[TRAIN]` section in the config.ini file must specify the following options for training 
the convolutional neural networks:

* `NETNAME` : string  
  Name of neural net architecture to train. Valid options are {'alexnet', 'VGG16'}
* `INPUT_SHAPE` : tuple  
  with three elements: height, width, and number of channels in input image.
  For example, for AlexNet this must be at least (227, 227, 3).
* `BASE_LEARNING_RATE` : float  
  Learning rate applied to layers for which pre-trained weights are loaded. This should be a very 
  small number, e.g. `1e-20`.
* `NEW_LAYER_LEARNING_RATE` : float
  Learning rate applied to layers for which weights and biases are initialized randomly, instead of 
  using pre-trained weights; those layers will be trained with this learning rate so it should be 
  smaller than a typical learning rate, since we are "fine tuning", but not as small as `BASE_LEARNING_RATE`  
* `NEW_LEARN_RATE_LAYERS` : list  
  of strings, which are layer names. The weights and biases of these layers will be initialized 
  randomly, instead of using pre-trained weights, and trained with the optimizer using `NEW_LAYER_LEARNING_RATE`
* `NUMBER_NETS_TO_TRAIN` : integer  
  Number of neural net "replicates"; each replicate will be initialized and trained the 
  full number of specified epochs 
* `EPOCHS` : integer  
  Number of training epochs.
* `RANDOM_SEED` : integer  
  Integer used to seed random number generator.
* `BATCH_SIZE` : integer  
   Number of training examples to include in a batch.
* `MODEL_SAVE_PATH` : string  
  Path to directory where checkpoint files should be saved.

The remaining options for the `[TRAIN]` section are optional:
* `DROPOUT_RATE` : float
  Between 0 and 1. Allows for specifying a dropout rate during training that is different for the default,
  which is 0.5.

Here is an example `[TRAIN]` section:
```ini
[TRAIN]
NETNAME = alexnet
INPUT_SHAPE = (227, 227, 3)
BASE_LEARNING_RATE = 1e-20
NEW_LAYER_LEARNING_RATE = 0.00001
NEW_LEARN_RATE_LAYERS = ['fc6', 'fc7', 'fc8']
NUMBER_NETS_TO_TRAIN = 5
EPOCHS = 10
RANDOM_SEED = 42
BATCH_SIZE = 64
MODEL_SAVE_PATH = ../data/checkpoints/config_efficient_10_epochs_models/
```
### `[TEST]` section
The `[TEST]` section provide the following options for testing the trained convolutional neural networks:

* `TEST_RESULTS_SAVE_PATH` : string  
  Path to directory where results of measuring accuracy on a test set should be saved.

Here is an example `[TEST]` section:

```ini
[TEST]
TEST_RESULTS_SAVE_PATH = ../results/config_efficient_10_epochs_models/
```
