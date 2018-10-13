import tensorflow as tf
from src.utils.home_dir import get_home_directory


def get_checkpoint_directory():
    path = get_home_directory()
    return path


def read_checkpoint_db():
    pass


def get_checkpoint(db):
    pass


def get_chekpoint_config(id_or_alias):
    return get_checkpoint_directory()
