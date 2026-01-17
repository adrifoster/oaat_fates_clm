"""Helper methods"""

import yaml


def get_config_file(file_name: str) -> dict:
    """Reads in a YAML config file to a dictionary

    Args:
        file_name (str): path to config file

    Returns:
        dict: dictionary output
    """

    with open(file_name, "r") as f:
        config = yaml.safe_load(f)

    return config
