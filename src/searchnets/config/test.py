"""class to represent test section of config.ini file """
import attr

from attr.validators import instance_of


@attr.s
class TestConfig:
    """class to represent [TEST] section of config.ini file

    Attributes
    ----------
    test_results_save_path : string
        Path to directory where results of measuring accuracy on a test set should be saved.
    """
    test_results_save_path = attr.ib(validator=instance_of(str))
