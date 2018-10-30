import tensorflow as tf
import sonnet as snt
from src.utils.anchors import generate_anchors_reference
import numpy as np
from src.models.base import TruncatedBaseNetwork
from src.models.fasterrcnn.rpn import RPN
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
        self._num_anchors = self._anchor_reference[0]

        self._rpn_cls_loss_weight = config.model.loss.rpn_cls_loss_weight
        self._rpn_reg_loss_weight = config.model.loss.rpn_reg_loss_weights

        self._rcnn_cls_loss_weight = config.model.loss.rcnn_cls_loss_weight
        self._rcnn_reg_loss_weight = config.model.loss.rcnn_reg_loss_weights

        self.losses_collection = ['fastercnn_losses']
        self.base_network = TruncatedBaseNetwork(config.model.base_network)

    def _build(self, image, gt_box=None, is_training=False):
        image.set_shape((None, None, 3))
        conv_feature_map = self.base_network(
            tf.expand_dims(image, 0), is_training)
        self._rpn = RPN(self._num_anchors, self._config.model.rpn,
                        debug=self._debug, seed=self._seed)
        if self._with_rcnn:
            self._rcnn = None
        image_shape = tf.shape(image)[0:2]
        variable_summaries(conv_feature_map, 'conv_feature_map', 'reduced')
        all_anchors = self._generate_anchors(tf.shape(conv_feature_map))

    def _generate_anchors(self, feature_map_shape):
        return {}
