import configparser


def get_config(config_fname):
    """parse config file

    Parameters
    ----------
    config_fname : str
        name of config file

    Returns
    -------
    config : dict
        dictionary with all configuration parameters
    """

    config = configparser.ConfigParser()
    config.read(config_fname)
    return config
