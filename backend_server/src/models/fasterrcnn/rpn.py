import sonnet as snt
import tensorflow as tf
from src.utils.vars import get_initializer


class RPN(snt.AbstractModule):
    def __init__(self, num_anchors, config, debug=False, seed=None, name='rpn'):
        super(RPN, self).__init__(name=name)
        self._num_anchors = num_anchors
        self._num_channels = config.num_channels
        self._kernel_shape = config.kernel_shape
        self._debug = debug
        self._seed = seed
        self._regularizer = tf.contrib.layers.l2_regularizer(
            scale=config.l2_regularization_scale)
        self._l1_sigma = config.l1_sigma
        self._rpn_initializer = get_initializer(
            config.rpn_initializer, seed=seed)
        self._cls_initializer = get_initializer(
            config.cls_initializer, seed=seed)
        self._bbox_initializer = get_initializer(
            config.bbox_initializer, seed=seed)

        self._config = config

    def _build(self, conv_feature_map, im_shape, all_anchors, gt_boxes=None, is_training=False):
        pass
