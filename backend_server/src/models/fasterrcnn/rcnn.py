import sonnet as snt
from src.utils.vars import get_initializer
import tensorflow as tf
from src.models.fasterrcnn.roi_pool import ROIPoolingLayer


class RCNN(snt.AbstractModule):
    def __init__(self, num_classes, config, debug=False, seed=None, name='rcnn'):
        super(RCNN, self).__init__(name=name)
        self._num_classes = num_classes
        self._layer_sizes = config.lay_sizes
        self._activation = ()
        self._dropout_keep_prob = config.dropout_keep_prob
        self._use_mean = config.use_mean
        self._variances = config.target_normalization_variances
        self._rcnn_initializer = get_initializer(
            config.rcnn_initializer, seed=seed)
        self._cls_initializer = get_initializer(
            config.cls_initializer, seed=seed)
        self._bbox_initializer = get_initializer(
            config.bbox_initializer, seed=seed)
        self.regularizer = tf.contrib.layers.l2_regularizer(
            scale=config.l2_regularization_scale)
        self._l1_sigma = config.l1_sigma
        self._debug = debug
        self._config = config
        self._seed = seed

    def _build(self, conv_feature_map, proposals, im_shape, base_network, gt_boxes=None, is_training=False):
        self._instantiate_layers()
        prediction_dict = {'_debug': {}}

    def _instantiate_layers(self):
        self._layer = [snt.Linear(layer_size, name='fc{}'.format(i), initializers={'w': self._rcnn_initializer}, regularizers={
                                  'w': self.regularizer}) for i, layer_size in enumerate(self._layer_sizes)]
        self._clssifier_layer = snt.Linear(self._num_classes + 1, name='fc_classifier', initializers={
                                           'w': self._cls_initializer}, regularizers={'w': self.regularizer})
        self._bbox_layer = snt.Linear(self._num_classes * 4, name='fc_bbox', initializers={
                                      'w': self._bbox_initializer}, regularizers={'w': self.regularizer})
        self._roi_pool = ROIPoolingLayer(self._config.roi, debug=self._debug)
