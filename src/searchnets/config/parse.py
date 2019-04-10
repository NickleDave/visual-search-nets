import ast
import configparser
from distutils.util import strtobool

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
                               save_acc_by_set_size_by_epoch)

    # ------------- unpack [DATA] section of config.ini file -----------------------------------------------------------
    train_dir = config['DATA']['TRAIN_DIR']
    train_size = int(config['DATA']['TRAIN_SIZE'])
    if config.has_option('DATA', 'VALIDATION_SIZE'):
        val_size = int(config['DATA']['VALIDATION_SIZE'])
    else:
        val_size = None
    if config.has_option('DATA', 'SET_SIZES'):
        set_sizes = ast.literal_eval(config['DATA']['SET_SIZES'])
    else:
        set_sizes = None
    gz_filename = config['DATA']['GZ_FILENAME']

    data_config = DataConfig(train_dir,
                             train_size,
                             gz_filename,
                             val_size,
                             set_sizes)

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
