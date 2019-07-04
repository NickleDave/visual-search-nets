"""
Invokes __main__ when the module is run as a script.
Example: python -m searchnets --help
The package is installed on the path by pip, so typing
`$ searchnets --help` would have the same effect (i.e., no need
to type the python -m)
"""
import os
import argparse

from .config import parse_config
from .data import data
from .train import train
from .test import test
from .learncurve import learning_curve


def _call_data(config):
    """helper function to call searchstims.data"""
    data(train_dir=config.data.train_dir,
         train_size=config.data.train_size,
         stim_types=config.data.stim_types,
         val_size=config.data.val_size,
         test_size=config.data.test_size,
         gz_filename=config.data.gz_filename,
         train_size_per_set_size=config.data.train_size_per_set_size,
         val_size_per_set_size=config.data.val_size_per_set_size,
         test_size_per_set_size=config.data.test_size_per_set_size,
         shard_train=config.data.shard_train,
         shard_size=config.data.shard_size)


def _call_train(config):
    """helper function to call searchstims.train"""
    train(gz_filename=config.data.gz_filename,
          net_name=config.train.net_name,
          number_nets_to_train=config.train.number_nets_to_train,
          input_shape=config.train.input_shape,
          new_learn_rate_layers=config.train.new_learn_rate_layers,
          new_layer_learning_rate=config.train.new_layer_learning_rate,
          epochs_list=config.train.epochs_list,
          batch_size=config.train.batch_size,
          random_seed=config.train.random_seed,
          model_save_path=config.train.model_save_path,
          base_learning_rate=config.train.base_learning_rate,
          freeze_trained_weights=config.train.freeze_trained_weights,
          dropout_rate=config.train.dropout_rate,
          save_acc_by_set_size_by_epoch=config.train.save_acc_by_set_size_by_epoch,
          use_val=config.train.use_val,
          val_step=config.train.val_step,
          patience=config.train.patience)


def _call_test(config, configfile):
    """helper function to call searchstims.test"""
    test(gz_filename=config.data.gz_filename,
         net_name=config.train.net_name,
         number_nets_to_train=config.train.number_nets_to_train,
         input_shape=config.train.input_shape,
         new_learn_rate_layers=config.train.new_learn_rate_layers,
         epochs_list=config.train.epochs_list,
         batch_size=config.train.batch_size,
         model_save_path=config.train.model_save_path,
         test_results_save_path=config.test.test_results_save_path,
         configfile=configfile)


def _call_learncurve(config):
    """helper function to call searchstims.learncurve"""
    learning_curve(gz_filename=config.data.gz_filename,
                   net_name=config.train.net_name,
                   number_nets_to_train=config.train.number_nets_to_train,
                   input_shape=config.train.input_shape,
                   new_learn_rate_layers=config.train.new_learn_rate_layers,
                   base_learning_rate=config.train.base_learning_rate,
                   new_layer_learning_rate=config.train.new_layer_learning_rate,
                   epochs_list=config.train.epochs_list,
                   train_size_list=config.learncurve.train_size_list,
                   batch_size=config.train.batch_size,
                   random_seed=config.train.random_seed,
                   model_save_path=config.train.model_save_path,
                   test_results_save_path=config.test.test_results_save_path,
                   dropout_rate=config.train.dropout_rate,
                   val_size=config.data.val_size)


def cli(command, configfile):
    """command-line interface
    Called by main() when user runs from the command-line by typing 'searchnets'

    Parameters
    ----------
    command : str
        Command to follow. One of {'train', 'test'}
            Train : train models using configuration defined in config file.
            Test : test accuracy of trained models using configuration defined in configfile.

    configfile : str
        Path to a `config.ini` file that defines the configuration.

    Returns
    -------
    None

    Examples
    --------
    >>> cli(command='train', config='./configs/quick_run_config.ini')

    Notes
    -----
    This function is not really meant to be run by the user, but has its own arguments
    to make it easier to test (instead of throwing everything into one 'main' function)
    """
    # get config first so we can know if we should save log, where to make results directory, etc.
    config = parse_config(configfile)

    if command == 'data':
        _call_data(config)

    elif command == 'train':
        _call_train(config)

    elif command == 'test':
        _call_test(config, configfile)

    elif command == 'all':
        _call_data(config)
        _call_train(config)
        _call_test(config, configfile)

    elif command == 'learncurve':
        _call_learncurve(config)


CHOICES = ['data', 'train', 'test', 'all', 'learncurve']


def get_parser():
    parser = argparse.ArgumentParser(description='searchnets command line interface',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('command', type=str, choices=CHOICES,
                        help=f"Command to run, one of: {CHOICES}\n"
                             "$ searchstims train ./configs/config_2018-12-17.ini")
    parser.add_argument('configfile', type=str,
                        help='name of config.ini file to use \n'
                             '$ searchstims train ./configs/config_2018-12-17.ini')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if not os.path.isfile(args.configfile):
        raise FileNotFoundError(
            f'specified config.ini file not found: {args.configfile}'
        )
    cli(command=args.command,
        configfile=args.configfile)


if __name__ == '__main__':
    main()
