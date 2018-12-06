import tensorflow as tf
import sonnet as snt
from src.utils.anchors import generate_anchors_reference
import numpy as np
from src.models.base import TruncatedBaseNetwork
from src.models.fasterrcnn.rpn import RPN
from src.models.fasterrcnn.rcnn import RCNN
from src.utils.vars import variable_summaries


class FasterRCNN(snt.AbstractModule):
    def __init__(self, config, name='fasterrcnn'):
        super(FasterRCNN, self).__init__(name=name)
        self._config = config
        self._num_classes = config.model.network.num_classes
        self._with_rcnn = config.model.network.with_rcnn
        self._debug = config.train.debug
        self._seed = config.train.seed
        self._anchor_base_size = config.model.anchors.base_size
        self._anchor_scales = np.array(config.model.anchors.scales)
        self._anchor_ratios = np.array(config.model.anchors.ratios)
        self._anchor_stride = config.model.anchors.stride
        self._anchor_reference = generate_anchors_reference(
            self._anchor_base_size, self._anchor_ratios, self._anchor_scales)
        print(self._anchor_reference)
        self._num_anchors = self._anchor_reference.shape[0]

        self._rpn_cls_loss_weight = config.model.loss.rpn_cls_loss_weight
        self._rpn_reg_loss_weight = config.model.loss.rpn_reg_loss_weights

        self._rcnn_cls_loss_weight = config.model.loss.rcnn_cls_loss_weight
        self._rcnn_reg_loss_weight = config.model.loss.rcnn_reg_loss_weights

        self.losses_collection = ['fastercnn_losses']
        self.base_network = TruncatedBaseNetwork(config.model.base_network)

    def _build(self, image, gt_boxes=None, is_training=False):
        image.set_shape((None, None, 3))
        conv_feature_map = self.base_network(
            tf.expand_dims(image, 0), is_training=is_training)
        self._rpn = RPN(self._num_anchors, self._config.model.rpn,
                        debug=self._debug, seed=self._seed)
        if self._with_rcnn:
            self._rcnn = RCNN(
                self._num_classes, self._config.model.rcnn, debug=self._debug, seed=self._seed)
        image_shape = tf.shape(image)[0:2]
        variable_summaries(conv_feature_map, 'conv_feature_map', 'reduced')
        all_anchors = self._generate_anchors(tf.shape(conv_feature_map))
        rpn_prediction = self._rpn(
            conv_feature_map, image_shape, all_anchors, gt_boxes=gt_boxes, is_training=is_training)
        prediction_dict = {'rpn_prediction': rpn_prediction}
        if self._with_rcnn:
            proposals = tf.stop_gradient(rpn_prediction['proposals'])
            classification_pred = self._rcnn(
                conv_feature_map, proposals, image_shape, self.base_network, gt_boxes=gt_boxes, is_training=is_training)
            prediction_dict['classification_prediction'] = classification_pred
        return prediction_dict

    def _generate_anchors(self, feature_map_shape):
        with tf.variable_scope('generate_anchors'):
            grid_width = feature_map_shape[2]
            grid_height = feature_map_shape[1]
            shift_x = tf.range(grid_width) * self._anchor_stride
            shift_y = tf.range(grid_height) * self._anchor_stride
            shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
            shift_x = tf.reshape(shift_x, [-1])
            shift_y = tf.reshape(shift_y, [-1])
            shifts = tf.stack([shift_x, shift_y, shift_x, shift_y], axis=0)
            shifts = tf.transpose(shifts)
            all_anchors = (np.expand_dims(self._anchor_reference,
                                          axis=0) + tf.expand_dims(shifts, axis=1))
            all_chors = tf.reshape(all_anchors, (-1, 4))
            return all_chors
