import tensorflow as tf
import json
import os
from src.datasets import get_dataset
from src.models import get_model


class PredictorNetwork(object):
    def __init__(self, config):
        if config.dataset.dir:
            class_labels_path = os.path.join(
                config.dataset.dir, 'classes.json')
            self.class_labels = json.load(tf.gfile.GFile(class_labels_path))
            print(self.class_labels)
        else:
            self.class_labels = None

        config.dataset.data_augmentation = None
        dataset_class = get_dataset(config.dataset.type)
        model_class = get_model(config.model.type)
        dataset = dataset_class(config)
        model = model_class(config)
