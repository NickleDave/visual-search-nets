import configparser

import attr
from attr.validators import instance_of, optional


@attr.s
class TrainConfig:
    """class to represent Train section of config.ini file

    gz_filename : str
        name of .gz file containing prepared data sets
    net_name : str
        name of convolutional neural net architecture to train.
        One of {'alexnet', 'VGG16'}
    number_nets_to_train : int
        number of training "replicates"
    input_shape : tuple
        with 3 elements, (rows, columns, channels)
        should be (227, 227, 3) for AlexNet
        and (224, 224, 3) for VGG16
    new_learn_rate_layers : list
        of layer names whose weights will be initialized randomly
        and then trained with the 'new_layer_learning_rate'.
    base_learning_rate : float
        Applied to layers with weights loaded from training the
        architecture on ImageNet. Should be a very small number
        so the trained weights don't change much.
    new_layer_learning_rate : float
        Applied to `new_learn_rate_layers'. Should be larger than
        `base_learning_rate` but still smaller than the usual
        learning rate for a deep net trained with SGD,
        e.g. 0.001 instead of 0.01
    epochs_list : list
        of training epochs. Replicates will be trained for each
        value in this list. Can also just be one value, but a list
        is useful if you want to test whether effects depend on
        number of training epochs.
    batch_size : int
        number of samples in a batch of training data
    random_seed : int
        to seed random number generator
    model_save_path : str
        path to directory where model checkpoints should be saved
    dropout_rate : float
        Probability that any unit in a layer will "drop out" during
        a training epoch, as a form of regularization. Default is 0.5.
    val_size : int
        number of samples in validation set. Default is None.
    """

    train_size = attr.ib(validator=instance_of(int))
    val_size = attr.ib(validator=optional(instance_of(int)), default=None)
    gz_filename = attr.ib(validator=instance_of(str))
    net_name = attr.ib(validator=instance_of(list))
    number_nets_to_train = attr.ib(validator=instance_of(int))
    input_shape = attr.ib(validator=instance_of(tuple))

    @input_shape.validator
    def check_input_shape(self, attribute, value):
        if len(value) != 3:
            raise ValueError(f"input shape should be 3-element tuple (rows, columns, channels) but got {value}")

    new_learn_rate_layers = attr.ib(validator=instance_of(list))

    @new_learn_rate_layers.validator
    def check_new_learn_rate_layers(self, attribute, value):
        for layer_name in value:
            if type(layer_name) != str:
                raise TypeError(f'new_learn_rate_layer names should be strings but got {layer_name}')

    base_learning_rate = attr.ib(validator=instance_of(float))
    new_layer_learning_rate = attr.ib(validator=instance_of(float))
    epochs_list = attr.ib(validator=instance_of(list))

    @epochs_list.validator
    def check_epochs_list(self, attribute, value):
        for ind, epochs in enumerate(value):
            if type(epochs) != int:
                raise TypeError('all values in epochs_list should be int but '
                                f'got type {type(epochs)} for element {ind}')

    batch_size = attr.ib(validator=instance_of(int))
    random_seed = attr.ib(validator=instance_of(int))
    dropout_rate = attr.ib(validator=instance_of(float), default=0.5)
    model_save_path = attr.ib(validator=instance_of(str))


@attr.s
class Config:
    train : TrainConfig


def parse_config(config_fname):
    """parse config.ini file
    Uses ConfigParser from Python standard library.

    Parameters
    ----------
    config_fname : str
        name of config.ini file

    Returns
    -------
    config : ConfigParser
        instance of ConfigParser; dictionary-like object
        with all configuration parameters
    """
    config = configparser.ConfigParser()
    config.read(config_fname)

    # ------------- unpack [TRAIN] section of config.ini file ----------------------------------------------------------
    net_name = config['TRAIN']['NETNAME']
    number_nets_to_train = int(config['TRAIN']['number_nets_to_train'])
    input_shape = ast.literal_eval(config['TRAIN']['INPUT_SHAPE'])
    new_learn_rate_layers = ast.literal_eval(config['TRAIN']['NEW_LEARN_RATE_LAYERS'])
    base_learning_rate = float(config['TRAIN']['BASE_LEARNING_RATE'])
    new_layer_learning_rate = float(config['TRAIN']['NEW_LAYER_LEARNING_RATE'])
    epochs_list = ast.literal_eval(config['TRAIN']['EPOCHS'])
    batch_size = int(config['TRAIN']['BATCH_SIZE'])
    random_seed = int(config['TRAIN']['RANDOM_SEED'])
    if config.has_option('TRAIN', 'DROPOUT_RATE'):
        dropout_rate = config['TRAIN']['DROPOUT_RATE']
    else:
        dropout_rate = 0.5
    model_save_path = config['TRAIN']['MODEL_SAVE_PATH']

    train_config = TrainConfig(train_size,
                               val_size,
                               gz_filename,
                               net_name,
                               number_nets_to_train,
                               input_shape,
                               base_learning_rate,
                               new_learn_rate_layers,
                               new_layer_learning_rate,
                               epochs_list,
                               batch_size,
                               random_seed,
                               model_save_path,
                               dropout_rate)

    # ------------- unpack [DATA] section of config.ini file -----------------------------------------------------------

    train_size = int(config['DATA']['TRAIN_SIZE'])
    if config.has_option('DATA', 'VALIDATION_SIZE'):
        val_size = int(config['DATA']['VALIDATION_SIZE'])
    else:
        val_size = None
    gz_filename = config['DATA']['GZ_FILENAME']


    config_obj = Config(train=train_config)

    return config
