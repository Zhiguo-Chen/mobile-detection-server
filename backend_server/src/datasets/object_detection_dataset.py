import tensorflow as tf
from src.datasets.base_dataset import BaseDataset

CONTEXT_FEATURES = {
    'image_raw': tf.FixedLenFeature([], tf.string),
    'filename': tf.FixedLenFeature([], tf.string),
    'width': tf.FixedLenFeature([], tf.int64),
    'height': tf.FixedLenFeature([], tf.int64),
    'depth': tf.FixedLenFeature([], tf.int64),
}

SEQUENCE_FEATURES = {
    'label': tf.VarLenFeature(tf.int64),
    'xmin': tf.VarLenFeature(tf.int64),
    'xmax': tf.VarLenFeature(tf.int64),
    'ymin': tf.VarLenFeature(tf.int64),
    'ymax': tf.VarLenFeature(tf.int64),
}


class ObjectDetectionDataset(BaseDataset):
    def __init__(self, config, name='object_detection_dataset', **kwargs):
        super(ObjectDetectionDataset, self).__init__(config, **kwargs)
        self._image_min_size = config.datasets.image_preprocessing.get(
            'min_size')
        self._image_max_size = config.datasets.image_preprocessing.get(
            'max_size')
        self._data_augmentation = config.dataset.data_augmentation or []
