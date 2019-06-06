import ast
import configparser
from distutils.util import strtobool
import os

from .classes import TrainConfig, DataConfig, TestConfig, LearnCurveConfig, Config


def parse_config(config_fname):
    """parse config.ini file
    Uses ConfigParser from Python standard library.

    Parameters
    ----------
    config_fname : str
        name of config.ini file

    Returns
    -------
    config_obj : Config
        instance of searchstims.config.classes.Config,
        attrs-based class that represents all configuration parameters
    """
    if not os.path.isfile(config_fname):
        raise FileNotFoundError(
            f'specified config.ini file not found: {configfile}'
        )

    config = configparser.ConfigParser()
    config.read(config_fname)

    # ------------- unpack [TRAIN] section of config.ini file ----------------------------------------------------------
    net_name = config['TRAIN']['NETNAME']
    number_nets_to_train = int(config['TRAIN']['number_nets_to_train'])
    input_shape = ast.literal_eval(config['TRAIN']['INPUT_SHAPE'])
    new_learn_rate_layers = ast.literal_eval(config['TRAIN']['NEW_LEARN_RATE_LAYERS'])
    base_learning_rate = float(config['TRAIN']['BASE_LEARNING_RATE'])
    new_layer_learning_rate = float(config['TRAIN']['NEW_LAYER_LEARNING_RATE'])

    if config.has_option('TRAIN', 'FREEZE_TRAINED_WEIGHTS'):
        freeze_trained_weights = bool(strtobool(config['TRAIN']['FREEZE_TRAINED_WEIGHTS']))
    else:
        freeze_trained_weights = False

    epochs_list = ast.literal_eval(config['TRAIN']['EPOCHS'])
    if type(epochs_list) == int:
        epochs_list = [epochs_list]

    batch_size = int(config['TRAIN']['BATCH_SIZE'])
    random_seed = int(config['TRAIN']['RANDOM_SEED'])

    if config.has_option('TRAIN', 'DROPOUT_RATE'):
        dropout_rate = config['TRAIN']['DROPOUT_RATE']
    else:
        dropout_rate = 0.5

    model_save_path = config['TRAIN']['MODEL_SAVE_PATH']

    if config.has_option('TRAIN', 'SAVE_ACC_BY_SET_SIZE_BY_EPOCH'):
        save_acc_by_set_size_by_epoch = bool(strtobool(config['TRAIN']['SAVE_ACC_BY_SET_SIZE_BY_EPOCH']))
    else:
        save_acc_by_set_size_by_epoch = False

    if config.has_option('TRAIN', 'USE_VAL'):
        use_val = bool(strtobool(config['TRAIN']['USE_VAL']))
    else:
        use_val = True

    if config.has_option('TRAIN', 'VAL_STEP'):
        val_step = int(config['TRAIN']['VAL_STEP'])
    else:
        val_step = None

    if config.has_option('TRAIN', 'PATIENCE'):
        patience = int(config['TRAIN']['PATIENCE'])
    else:
        patience = None

    train_config = TrainConfig(net_name,
                               number_nets_to_train,
                               input_shape,
                               new_learn_rate_layers,
                               new_layer_learning_rate,
                               epochs_list,
                               batch_size,
                               random_seed,
                               model_save_path,
                               base_learning_rate,
                               freeze_trained_weights,
                               dropout_rate,
                               save_acc_by_set_size_by_epoch,
                               use_val,
                               val_step,
                               patience)

    # ------------- unpack [DATA] section of config.ini file -----------------------------------------------------------
    train_dir = config['DATA']['TRAIN_DIR']
    train_size = int(config['DATA']['TRAIN_SIZE'])

    if config.has_option('DATA', 'STIM_TYPES'):
        stim_types = ast.literal_eval(config['DATA']['STIM_TYPES'])
    else:
        stim_types = None

    if config.has_option('DATA', 'VALIDATION_SIZE'):
        val_size = int(config['DATA']['VALIDATION_SIZE'])
    else:
        val_size = None
    if config.has_option('DATA', 'TEST_SIZE'):
        test_size = int(config['DATA']['TEST_SIZE'])
    else:
        test_size = None

    if config.has_option('DATA', 'SET_SIZES'):
        set_sizes = ast.literal_eval(config['DATA']['SET_SIZES'])
    else:
        set_sizes = None
    gz_filename = config['DATA']['GZ_FILENAME']

    if config.has_option('DATA', 'TRAIN_SIZE_PER_SET_SIZE'):
        train_size_per_set_size = ast.literal_eval(config['DATA']['TRAIN_SIZE_PER_SET_SIZE'])
    else:
        train_size_per_set_size = None
    if config.has_option('DATA', 'VAL_SIZE_PER_SET_SIZE'):
        val_size_per_set_size = ast.literal_eval(config['DATA']['VAL_SIZE_PER_SET_SIZE'])
    else:
        val_size_per_set_size = None
    if config.has_option('DATA', 'TEST_SIZE_PER_SET_SIZE'):
        test_size_per_set_size = ast.literal_eval(config['DATA']['TEST_SIZE_PER_SET_SIZE'])
    else:
        test_size_per_set_size = None

    if config.has_option('DATA', 'SHARD_TRAIN'):
        shard_train = bool(strtobool(config['DATA']['SHARD_TRAIN']))
    else:
        shard_train = False
    if config.has_option('DATA', 'SHARD_SIZE'):
        shard_size = int(config['DATA']['SHARD_SIZE'])
    else:
        if shard_train:
            raise ValueError('shard_train set to True inf config.ini file but shard_size not specified')
        shard_size = None

    data_config = DataConfig(train_dir,
                             train_size,
                             gz_filename,
                             stim_types,
                             val_size,
                             test_size,
                             set_sizes,
                             train_size_per_set_size,
                             val_size_per_set_size,
                             test_size_per_set_size,
                             shard_train,
                             shard_size)

    # ------------- unpack [TEST] section of config.ini file -----------------------------------------------------------
    test_results_save_path = config['TEST']['TEST_RESULTS_SAVE_PATH']

    test_config = TestConfig(test_results_save_path)

    # ------------- unpack [TRAIN] section of config.ini file ----------------------------------------------------------
    if config.has_section('LEARNCURVE'):
        train_size_list = ast.literal_eval(config['LEARNCURVE']['TRAIN_SIZE_LIST'])
        learncurve_config = LearnCurveConfig(train_size_list)
    else:
        learncurve_config = None

    # ------------- make actual config object --------------------------------------------------------------------------
    config_obj = Config(train_config, data_config, test_config, learncurve_config)

    return config_obj
