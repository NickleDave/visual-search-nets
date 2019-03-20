"""
Invokes __main__ when the module is run as a script.
Example: python -m searchnets --help
The package is installed on the path by pip, so typing
`$ searchnets --help` would have the same effect (i.e., no need
to type the python -m)
"""
import argparse

from .config import parse_config
from .data import data
from .train import train
from .test import test


def cli(command, configfile):
    """command-line interface
    Called by main() when user runs ram from the command-line by typing 'searchnets'

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
        train_size = int(config['DATA']['TRAIN_SIZE'])
        if config.has_option('DATA', 'VALIDATION_SIZE'):
            val_size = int(config['DATA']['VALIDATION_SIZE'])
        else:
            val_size = None
        gz_filename = config['DATA']['GZ_FILENAME']

        data(train_dir=config['DATA']['TRAIN_DIR'],
             train_size=train_size,
             val_size=val_size,
             gz_filename=gz_filename
             )
    elif command == 'train':
        train(config)
    elif command == 'test':
        test(config)
    elif command == 'all':
        data(config)
        train(config)
        test(config)


def get_parser():
    parser = argparse.ArgumentParser(description='main script',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('command', type=str, choices=['data', 'train', 'test', 'all'],
                        help="Command to run, either 'data', 'train', 'test', or 'all'\n"
                             "$ searchstims train ./configs/config_2018-12-17.ini")
    parser.add_argument('configfile', type=str,
                        help='name of config.ini file to use \n'
                             '$ searchstims train ./configs/config_2018-12-17.ini')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    cli(command=args.command,
        configfile=args.configfile)


if __name__ == '__main__':
    main()
