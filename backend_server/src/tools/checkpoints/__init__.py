import tensorflow as tf
import os
from src.utils.home_dir import get_home_directory
import json
from src.utils.config import get_config

CHECKPOINTS_DIRECTORY = 'storage'
CHECKPOINTS_PATH = 'checkpoints'
CHECKPOINTS_INDEX = 'checkpoints.json'


def get_checkpoints_directory():
    path = os.path.join(get_home_directory(),
                        CHECKPOINTS_DIRECTORY, CHECKPOINTS_PATH)
    return path


def read_checkpoint_db():
    path = os.path.join(get_checkpoints_directory(), CHECKPOINTS_INDEX)
    if not os.path.exists(path):
        return {'checkpoints': {}}
    # with open(path) as f:
    #     index = json.load(f)
    index = json.load(tf.gfile.GFile(path))
    return index


def get_checkpoint_path(checkpoint_id):
    return os.path.join(get_checkpoints_directory(), checkpoint_id)


def get_checkpoint(db, id_or_alias):
    remote_checkpoints = sorted([c for c in db['checkpoints'] if c['source'] == 'remote'],
                                key=lambda c: c['created_at'], reverse=True)
    for cp in remote_checkpoints:
        if cp['id'] == id_or_alias or cp['alias'] == id_or_alias:
            return cp
    return {}


def get_chekpoint_config(id_or_alias):
    db = read_checkpoint_db()
    checkpoint = get_checkpoint(db, id_or_alias)
    if 'id' not in checkpoint or checkpoint['id'] is None:
        raise ValueError('checkpoint not found')
    checkpoint_path = get_checkpoint_path(checkpoint['id'])
    config = get_config(os.path.join(checkpoint_path, 'config.yml'))
    config.dataset.dir = checkpoint_path
    config.train.job_dir = get_checkpoints_directory()
    return config
