import tensorflow as tf
import numpy as np
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
            print('class_labels')
            print(self.class_labels)
        else:
            self.class_labels = None

        config.dataset.data_augmentation = None
        dataset_class = get_dataset(config.dataset.type)
        model_class = get_model(config.model.type)
        dataset = dataset_class(config)
        model = model_class(config)

        gragh = tf.Graph()
        self.session = tf.Session(graph=gragh)
        with gragh.as_default():
            self.image_placeholder = tf.placeholder(
                tf.float32, (None, None, 3)
            )
            print(self.image_placeholder)
            image_tf, _, process_meta = dataset.preprocess(
                self.image_placeholder)
            pred_dict = model(image_tf)
            if config.train.job_dir:
                job_dir = config.train.job_dir
                if config.train.run_name:
                    job_dir = os.path.join(job_dir, config.train.run_name)
                ckpt = tf.train.get_checkpoint_state(job_dir)
                if not ckpt or not ckpt.all_model_checkpoint_paths:
                    raise ValueError(
                        'Could not find checkpoint in {}'.format(job_dir))
                ckpt = ckpt.all_model_checkpoint_paths[-1]
                saver = tf.train.Saver(sharded=True, allow_empty=True)
                saver.restore(self.session, ckpt)
                tf.logging.info('Load checkpoint')
                if config.model.type == 'fasterrcnn':
                    cls_prediciton = pred_dict['classification_prediction']
                    objects_tf = cls_prediciton['objects']
                    objects_labels_tf = cls_prediciton['labels']
                    objects_labels_prob_tf = cls_prediciton['probs']

            self.fetches = {
                'objects': objects_tf,
                'labels': objects_labels_tf,
                'probs': objects_labels_prob_tf,
                'scale_factor': process_meta['scale_factor']
            }

    def predict_image(self, image):
        fetched = self.session.run(self.fetches, feed_dict={
            self.image_placeholder: np.array(image)
        })
        objects = fetched['objects']
        labels = fetched['labels'].tolist()
        probs = fetched['probs'].tolist()
        scale_factor = fetched['scale_factor']
        if self.class_labels is not None:
            labels = [self.class_labels[label] for label in labels]
        if isinstance(scale_factor, tuple):
            objects /= [scale_factor[1], scale_factor[0],
                        scale_factor[1], scale_factor[0]]
        else:
            objects /= scale_factor

        objects = [[round(coord) for coord in obj] for obj in objects.tolist()]
        predictions = sorted([
            {
                'bbox': obj,
                'label': label,
                'prob': round(prob, 4)
            } for obj, label, prob in zip(objects, labels, probs)
        ], key=lambda x: x['prob'], reverse=True)
        return predictions
