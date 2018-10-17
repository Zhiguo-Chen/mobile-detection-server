import yaml
import tensorflow as tf
from easydict import EasyDict
from src.models import get_model
import os
import inspect

REPLACE_KEY = '_replace'


def get_config(path, override_params=None):
    custom_config = load_config_file(path)
    model_class = get_model(custom_config['model']['type'])
    model_base_config = get_base_config(model_class)
    config = get_model_config(
        model_base_config, custom_config, override_params)
    return config


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


def get_base_config(model_class, base_config_filename='base_config.yml'):
    path = inspect.getfile(model_class)
    print(path, ' >>>>')
    config_path = os.path.join(os.path.dirname(path), base_config_filename)
    return load_config_file([config_path])


def get_model_config(base_config, custom_config, override_params):
    config = EasyDict(base_config.copy())
    if custom_config:
        config = merge_into(custom_config, base_config, overwrite=True)
    return cleanup_config(config)


def cleanup_config(config):
    cleanup_keys = [REPLACE_KEY]
    for cleanup_key in cleanup_keys:
        config.pop(cleanup_key, None)
    for config_key in config:
        if isinstance(config[config_key], dict):
            cleanup_config(config[config_key])
    return config
