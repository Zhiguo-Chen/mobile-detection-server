import sonnet as snt
import tensorflow as tf
from src.utils.vars import (
    get_initializer, get_activation_function, variable_summaries)
from sonnet.python.modules.conv import Conv2D
from .rpn_proposal import RPNProposal


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

        self._rpn_activation = get_activation_function(
            config.activation_function)

        self._config = config

    def _build(self, conv_feature_map, im_shape, all_anchors, gt_boxes=None, is_training=False):
        self._instantiate_layers()
        self._proposal = RPNProposal(
            self._num_anchors, self._config.proposals, debug=self._debug)
        prediction_dict = {}
        rpn_conv_feature = self._rpn(conv_feature_map)
        rpn_feature = self._rpn_activation(rpn_conv_feature)
        rpn_cls_score_original = self._rpn_cls(rpn_feature)
        rpn_bbox_pred_original = self._rpn_bbox(rpn_feature)
        rpn_cls_score = tf.reshape(rpn_cls_score_original, [-1, 2])
        rpn_cls_prob = tf.nn.softmax(rpn_cls_score)
        prediction_dict['rpn_cls_prob'] = rpn_cls_prob
        prediction_dict['rpn_cls_score'] = rpn_cls_score
        rpn_bbox_pred = tf.reshape(rpn_bbox_pred_original, [-1, 4])
        prediction_dict['rpn_bbox_pred'] = rpn_bbox_pred
        proposal_prediction = self._proposal(
            rpn_cls_prob, rpn_bbox_pred, all_anchors, im_shape)
        prediction_dict['proposals'] = proposal_prediction['proposals']
        prediction_dict['scores'] = proposal_prediction['scores']
        variable_summaries(prediction_dict['scores'], 'rpn_scores', 'reduced')
        variable_summaries(rpn_cls_prob, 'rpn_cls_prob', 'reduced')
        variable_summaries(rpn_bbox_pred, 'rpn_bbox_pred', 'reduced')
        return prediction_dict

    def _instantiate_layers(self):
        self._rpn = Conv2D(output_channels=self._num_channels, kernel_shape=self._kernel_shape, initializers={
                           'w': self._rpn_initializer}, regularizers={'w': self._regularizer}, name='conv')
        self._rpn_cls = Conv2D(output_channels=self._num_anchors * 2, kernel_shape=[1, 1], initializers={
                               'w': self._cls_initializer}, regularizers={'w': self._regularizer}, padding='VALID', name='cls_conv')
        self._rpn_bbox = Conv2D(output_channels=self._num_anchors * 4, kernel_shape=[
                                1, 1], initializers={'w': self._bbox_initializer}, regularizers={'w': self._regularizer}, padding='VALID', name='bbox_conv')
