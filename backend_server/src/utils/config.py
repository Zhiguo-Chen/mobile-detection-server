import yaml
import tensorflow as tf
from easydict import EasyDict

REPLACE_KEY = '_replace'


def get_config(path):
    custom_config = load_config_file(path)
    return {}


def load_config_file(filename_or_filenames):
    if(isinstance(filename_or_filenames, list)) or (isinstance(filename_or_filenames, tuple)):
        filenames = filename_or_filenames
    else:
        filenames = [filename_or_filenames]
        config = EasyDict({})
        for filename in filenames:
            with tf.gfile.GFile(filename) as f:
                new_config = EasyDict(yaml.load(f))
            config = merge_into(new_config, config, overwrite=True)
        return config


def merge_into(new_config, base_config, overwrite=False):
    if type(new_config) is not EasyDict:
        return
    else:
        for key, value in new_config.items():
            if isinstance(value, dict):
                if should_replace(new_config, base_config, key):
                    base_config[key] = value
                else:
                    base_config[key] = merge_into(
                        new_config[key], base_config.get(key, EasyDict({})), overwrite)
            else:
                if base_config.get(key) is None:
                    base_config[key] = value
                elif overwrite:
                    base_config[key] = value

    return base_config


def should_replace(new_config, base_config, key):
    try:
        base_config_replace = new_config[key][REPLACE_KEY]
    except KeyError:
        base_config_replace = None

    try:
        new_config_replace = new_config[key][REPLACE_KEY]
    except KeyError:
        new_config_replace = None

    if new_config_replace:
        return True
    elif new_config_replace is None and base_config_replace:
        return True

    return False


def get_model(model_name):
    pass
