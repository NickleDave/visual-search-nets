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
from .data import split
from .train import train
from .test import test


def _call_split(config):
    """helper function to call searchstims.data.split"""
    split(csv_file_in=config.data.csv_file_in,
          train_size=config.data.train_size,
          dataset_type=config.data.dataset_type,
          csv_file_out=config.data.csv_file_out,
          stim_types=config.data.stim_types,
          val_size=config.data.val_size,
          test_size=config.data.test_size,
          train_size_per_set_size=config.data.train_size_per_set_size,
          val_size_per_set_size=config.data.val_size_per_set_size,
          test_size_per_set_size=config.data.test_size_per_set_size)


def _call_train(config):
    """helper function to call searchstims.train"""
    train(csv_file=config.data.csv_file_out,
          dataset_type=config.data.dataset_type,
          net_name=config.train.net_name,
          number_nets_to_train=config.train.number_nets_to_train,
          epochs_list=config.train.epochs_list,
          batch_size=config.train.batch_size,
          random_seed=config.train.random_seed,
          root=config.data.root,
          pad_size=config.data.pad_size,
          save_path=config.train.save_path,
          method=config.train.method,
          num_classes=config.data.num_classes,
          learning_rate=config.train.learning_rate,
          new_learn_rate_layers=config.train.new_learn_rate_layers,
          new_layer_learning_rate=config.train.new_layer_learning_rate,
          base_learning_rate=config.train.base_learning_rate,
          freeze_trained_weights=config.train.freeze_trained_weights,
          loss_func=config.train.loss_func,
          optimizer=config.train.optimizer,
          use_val=config.train.use_val,
          val_epoch=config.train.val_epoch,
          summary_step=config.train.summary_step,
          patience=config.train.patience,
          checkpoint_epoch=config.train.checkpoint_epoch,
          save_acc_by_set_size_by_epoch=config.train.save_acc_by_set_size_by_epoch,
          num_workers=config.train.num_workers,
          data_parallel=config.train.data_parallel)


def _call_test(config, configfile):
    """helper function to call searchstims.test"""
    test(csv_file=config.data.csv_file_out,
         dataset_type=config.data.dataset_type,
         net_name=config.train.net_name,
         number_nets_to_train=config.train.number_nets_to_train,
         epochs_list=config.train.epochs_list,
         batch_size=config.train.batch_size,
         restore_path=config.train.save_path,
         test_results_save_path=config.test.test_results_save_path,
         configfile=configfile,
         random_seed=config.train.random_seed,
         root=config.data.root,
         pad_size=config.data.pad_size,
         num_classes=config.data.num_classes,
         num_workers=config.train.num_workers,
         data_parallel=config.train.data_parallel)


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

    if command == 'split':
        _call_split(config)

    elif command == 'train':
        _call_train(config)

    elif command == 'test':
        _call_test(config, configfile)

    elif command == 'all':
        _call_split(config)
        _call_train(config)
        _call_test(config, configfile)


CHOICES = ['split',
           'train',
           'test',
           'all',
           ]


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
