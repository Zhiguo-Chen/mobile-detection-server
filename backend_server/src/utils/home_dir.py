import tensorflow as tf
import os

DEFAULT_DIR = 'checkpoints'


def get_home_directory():
    path = os.path.abspath('')
    return path
