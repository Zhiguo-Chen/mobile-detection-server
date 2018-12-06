import sonnet as snt
from src.utils.vars import get_initializer, variable_summaries, get_activation_function
import tensorflow as tf
from src.models.fasterrcnn.roi_pool import ROIPoolingLayer
from src.models.fasterrcnn.rcnn_proposal import RCNNProposal


class RCNN(snt.AbstractModule):
    def __init__(self, num_classes, config, debug=False, seed=None, name='rcnn'):
        super(RCNN, self).__init__(name=name)
        self._num_classes = num_classes
        self._layer_sizes = config.layer_sizes
        self._activation = get_activation_function(config.activation_function)
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
        roi_prediction = self._roi_pool(proposals, conv_feature_map, im_shape)
        pooled_features = roi_prediction['roi_pool']
        features = base_network._build_tail(
            pooled_features, is_training=is_training)
        if self._use_mean:
            features = tf.reduce_mean(features, [1, 2])
        flatten_features = tf.contrib.layers.flatten(features)
        net = tf.identity(flatten_features)
        if is_training:
            net = tf.nn.dropout(net, keep_prob=self._dropout_keep_prob)
        for i, layer in enumerate(self._layers):
            net = layer(net)
            variable_summaries(
                net, 'fc_{}_preactivationout'.format(i), 'reduced')
            net = self._activation(net)
            variable_summaries(
                net, 'fc_{}_out'.format(i), 'reduced')
        cls_score = self._classifier_layer(net)
        cls_prob = tf.nn.softmax(cls_score, axis=1)
        bbox_offsets = self._bbox_layer(net)
        prediction_dict['rcnn'] = {
            'cls_score': cls_score,
            'cls_prob': cls_prob,
            'bbox_offsets': bbox_offsets
        }
        proposals_pred = self._rcnn_proposal(
            proposals, bbox_offsets, cls_prob, im_shape)
        prediction_dict['objects'] = proposals_pred['objects']
        prediction_dict['labels'] = proposals_pred['proposal_label']
        prediction_dict['probs'] = proposals_pred['proposal_label_prob']
        variable_summaries(cls_prob, 'cls_prob', 'reduced')
        variable_summaries(bbox_offsets, 'bbox_offsets', 'reduced')

        return prediction_dict

    def _instantiate_layers(self):
        self._layers = [snt.Linear(layer_size, name='fc{}'.format(i), initializers={'w': self._rcnn_initializer}, regularizers={
            'w': self.regularizer}) for i, layer_size in enumerate(self._layer_sizes)]
        self._classifier_layer = snt.Linear(self._num_classes + 1, name='fc_classifier', initializers={
            'w': self._cls_initializer}, regularizers={'w': self.regularizer})
        self._bbox_layer = snt.Linear(self._num_classes * 4, name='fc_bbox', initializers={
                                      'w': self._bbox_initializer}, regularizers={'w': self.regularizer})
        self._roi_pool = ROIPoolingLayer(self._config.roi, debug=self._debug)
        self._rcnn_proposal = RCNNProposal(
            self._num_classes, self._config.proposals, variances=self._variances)
