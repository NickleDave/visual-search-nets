import ast
import configparser
from distutils.util import strtobool
import os

import attr
from attr.validators import instance_of

from .data import DataConfig
from .test import TestConfig
from .train import TrainConfig


@attr.s
class Config:
    """class to represent all sections of config.ini file

    Attributes
    ----------
    train: TrainConfig
        represents [TRAIN] section
    data: DataConfig
        represents [DATA] section
    test: TestConfig
        represents [TEST] section
    """
    train = attr.ib(validator=instance_of(TrainConfig))
    data = attr.ib(validator=instance_of(DataConfig))
    test = attr.ib(validator=instance_of(TestConfig))


def _str_to_int_or_float(val):
    """helper function that tries to cast a str to int and if that fails then tries casting to float"""
    try:
        val = int(val)
    except ValueError:
        val = float(val)
    return val


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
            f'specified config.ini file not found: {config_fname}'
        )

    config = configparser.ConfigParser()
    config.read(config_fname)

    # ------------- unpack [DATA] section of config.ini file -----------------------------------------------------------
    csv_file_in = config['DATA']['CSV_FILE_IN']
    train_size = _str_to_int_or_float(config['DATA']['TRAIN_SIZE'])

    if config.has_option('DATA', 'DATASET_TYPE'):
        dataset_type = config['DATA']['DATASET_TYPE']
    else:
        dataset_type = 'searchstims'

    if config.has_option('DATA', 'NUM_CLASSES'):
        num_classes = int(config['DATA']['NUM_CLASSES'])
    else:
        num_classes = 2

    if config.has_option('DATA', 'ROOT'):
        root = config['DATA']['ROOT']
    else:
        root = None

    if config.has_option('DATA', 'PAD_SIZE'):
        pad_size = int(config['DATA']['PAD_SIZE'])
    else:
        pad_size = 500

    if config.has_option('DATA', 'CSV_FILE_OUT'):
        csv_file_out = config['DATA']['CSV_FILE_OUT']
    else:
        csv_file_out = None

    if config.has_option('DATA', 'STIM_TYPES'):
        stim_types = ast.literal_eval(config['DATA']['STIM_TYPES'])
    else:
        stim_types = None

    if config.has_option('DATA', 'VALIDATION_SIZE'):
        val_size = _str_to_int_or_float(config['DATA']['VALIDATION_SIZE'])
    else:
        val_size = None
    if config.has_option('DATA', 'TEST_SIZE'):
        test_size = _str_to_int_or_float(config['DATA']['TEST_SIZE'])
    else:
        test_size = None

    if config.has_option('DATA', 'SET_SIZES'):
        set_sizes = ast.literal_eval(config['DATA']['SET_SIZES'])
    else:
        set_sizes = None

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

    data_config = DataConfig(csv_file_in,
                             train_size,
                             dataset_type,
                             num_classes,
                             root,
                             pad_size,
                             csv_file_out,
                             stim_types,
                             val_size,
                             test_size,
                             set_sizes,
                             train_size_per_set_size,
                             val_size_per_set_size,
                             test_size_per_set_size)

    # ------------- unpack [TRAIN] section of config.ini file ----------------------------------------------------------
    # do some validation first
    if config.has_option('TRAIN', 'METHOD'):
        if config['TRAIN']['METHOD'] == 'transfer':
            if config.has_option('TRAIN', 'LEARNING_RATE'):
                raise ValueError('option "LEARNING_RATE" should only be specified when METHOD is "initialize"')
        if config['TRAIN']['METHOD'] == 'initialize':
            if any(
                [config.has_option('TRAIN', transfer_option)
                 for transfer_option in ['NEW_LEARN_RATE_LAYERS',
                                         'NEW_LAYER_LEARNING_RATE',
                                         'BASE_LEARNING_RATE',
                                         'FREEZE_TRAINED_WEIGHTS']]
            ):
                raise ValueError('METHOD specified as "initialize" but options for "transfer" were specified')

    # now actually unpack
    net_name = config['TRAIN']['NETNAME']
    number_nets_to_train = int(config['TRAIN']['number_nets_to_train'])
    batch_size = int(config['TRAIN']['BATCH_SIZE'])
    random_seed = int(config['TRAIN']['RANDOM_SEED'])
    save_path = config['TRAIN']['SAVE_PATH']

    if config.has_option('TRAIN', 'METHOD'):
        method = config['TRAIN']['METHOD']
    else:
        method = 'transfer'

    if config.has_option('TRAIN', 'LEARNING_RATE'):
        learning_rate = float(config['TRAIN']['LEARNING_RATE'])
    else:
        learning_rate = 0.001

    if config.has_option('TRAIN', 'NEW_LEARN_RATE_LAYERS'):
        new_learn_rate_layers = ast.literal_eval(config['TRAIN']['NEW_LEARN_RATE_LAYERS'])
    else:
        new_learn_rate_layers = ['fc8']
    if config.has_option('TRAIN', 'NEW_LAYER_LEARNING_RATE'):
        new_layer_learning_rate = float(config['TRAIN']['NEW_LAYER_LEARNING_RATE'])
    else:
        new_layer_learning_rate = 0.001
    if config.has_option('TRAIN', 'BASE_LEARNING_RATE'):
        base_learning_rate = float(config['TRAIN']['BASE_LEARNING_RATE'])
    else:
        base_learning_rate = 1e-20
    if config.has_option('TRAIN', 'FREEZE_TRAINED_WEIGHTS'):
        freeze_trained_weights = bool(strtobool(config['TRAIN']['FREEZE_TRAINED_WEIGHTS']))
    else:
        freeze_trained_weights = False

    epochs_list = ast.literal_eval(config['TRAIN']['EPOCHS'])
    if type(epochs_list) == int:
        epochs_list = [epochs_list]

    if config.has_option('TRAIN', 'LOSS_FUNC'):
        loss_func = config['TRAIN']['LOSS_FUNC']
    else:
        loss_func = 'CE'

    if config.has_option('TRAIN', 'OPTIMIZER'):
        optimizer = config['TRAIN']['OPTIMIZER']
    else:
        optimizer = 'SGD'

    if config.has_option('TRAIN', 'SAVE_ACC_BY_SET_SIZE_BY_EPOCH'):
        save_acc_by_set_size_by_epoch = bool(strtobool(config['TRAIN']['SAVE_ACC_BY_SET_SIZE_BY_EPOCH']))
    else:
        save_acc_by_set_size_by_epoch = False

    if config.has_option('TRAIN', 'USE_VAL'):
        use_val = bool(strtobool(config['TRAIN']['USE_VAL']))
    else:
        use_val = True

    if config.has_option('TRAIN', 'VAL_EPOCH'):
        val_epoch = int(config['TRAIN']['VAL_EPOCH'])
    else:
        val_epoch = None

    if config.has_option('TRAIN', 'SUMMARY_STEP'):
        summary_step = int(config['TRAIN']['SUMMARY_STEP'])
    else:
        summary_step = None

    if config.has_option('TRAIN', 'PATIENCE'):
        patience = int(config['TRAIN']['PATIENCE'])
    else:
        patience = None

    if config.has_option('TRAIN', 'CHECKPOINT_EPOCH'):
        checkpoint_epoch = int(config['TRAIN']['CHECKPOINT_EPOCH'])
    else:
        checkpoint_epoch = None

    if config.has_option('TRAIN', 'NUM_WORKERS'):
        num_workers = int(config['TRAIN']['NUM_WORKERS'])
    else:
        num_workers = 4

    if config.has_option('TRAIN', 'DATA_PARALLEL'):
        data_parallel = bool(strtobool(config['TRAIN']['DATA_PARALLEL']))
    else:
        data_parallel = False

    train_config = TrainConfig(net_name=net_name,
                               number_nets_to_train=number_nets_to_train,
                               epochs_list=epochs_list,
                               batch_size=batch_size,
                               random_seed=random_seed,
                               save_path=save_path,
                               method=method,
                               learning_rate=learning_rate,
                               new_learn_rate_layers=new_learn_rate_layers,
                               new_layer_learning_rate=new_layer_learning_rate,
                               base_learning_rate=base_learning_rate,
                               freeze_trained_weights=freeze_trained_weights,
                               loss_func=loss_func,
                               optimizer=optimizer,
                               save_acc_by_set_size_by_epoch=save_acc_by_set_size_by_epoch,
                               use_val=use_val,
                               val_epoch=val_epoch,
                               summary_step=summary_step,
                               patience=patience,
                               checkpoint_epoch=checkpoint_epoch,
                               num_workers=num_workers,
                               data_parallel=data_parallel)

    # ------------- unpack [TEST] section of config.ini file -----------------------------------------------------------
    test_results_save_path = config['TEST']['TEST_RESULTS_SAVE_PATH']

    test_config = TestConfig(test_results_save_path)

    # ------------- make actual config object --------------------------------------------------------------------------
    config_obj = Config(train_config, data_config, test_config)

    return config_obj
