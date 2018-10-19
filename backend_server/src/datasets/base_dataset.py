import os
import tensorflow as tf
import sonnet as snt


class BaseDataset(snt.AbstractModule):
    def __init__(self, config, **kwargs):
        super(BaseDataset, self).__init__(**kwargs)
        self._dataset_dir = config.dataset.dir
        self._num_epochs = config.train.num_epochs
        self._batch_size = config.train.batch_size
        self._split = config.dataset.split
        self._random_shuffle = config.train.random_shuffle
        self._seed = config.train.seed
        self._fixed_resize = (
            'fixed_height' in config.dataset.image_preprocessing and
            'fixed_width' in config.dataset.image_preprocessing
        )
        if self._fixed_resize:
            self._image_fixed_height = (
                config.dataset.image_preprocessing.fixed_height
            )
            self._image_fixed_width = (
                config.dataset.image_preprocessing.fixed_width
            )

        self._total_queue_ops = 20

    def _build(self):
        pass
