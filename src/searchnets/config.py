import configparser


def parse_config(config_fname):
    """parse config.ini file
    Uses ConfigParser from Python standard library.

    Parameters
    ----------
    config_fname : str
        name of config file

    Returns
    -------
    config : ConfigParser
        instance of ConfigParser; dictionary-like object
        with all configuration parameters
    """
    config = configparser.ConfigParser()
    config.read(config_fname)
    return config
