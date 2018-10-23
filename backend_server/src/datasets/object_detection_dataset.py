import tensorflow as tf
from src.datasets.base_dataset import BaseDataset
from src.utils.image import resize_image

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
        self._image_min_size = config.dataset.image_preprocessing.get(
            'min_size')
        self._image_max_size = config.dataset.image_preprocessing.get(
            'max_size')
        self._data_augmentation = config.dataset.data_augmentation or []

    def preprocess(self, image, bboxes=None):
        image, scale_factor = self._resize_image(image)
        return image, {'scale_factor': scale_factor}

    def _resize_image(self, image, bboxes=None):
        resized_image = resize_image(
            image, bboxes=bboxes, min_size=self._image_min_size, max_size=self._image_max_size)
        return resized_image['image'], resized_image['scale_factor']
