import tensorflow as tf
from .base_network import BaseNetwork
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_utils, resnet_v1

DEFAULT_ENDPOINTS = {
    'resnet_v1_50': 'block3',
    'resnet_v1_101': 'block3',
    'resnet_v1_152': 'block3',
    'resnet_v2_50': 'block3',
    'resnet_v2_101': 'block3',
    'resnet_v2_152': 'block3',
    'vgg_16': 'conv5/conv5_3',
}


class TruncatedBaseNetwork(BaseNetwork):
    def __init__(self, config, name='truncated_base_network', **kwargs):
        super(TruncatedBaseNetwork, self).__init__(config, name=name)

        self._endpoint = (
            config.endpoint or DEFAULT_ENDPOINTS[config.architecture]
        )
        self._scope_endpoint = '{}/{}/{}'.format(
            self.module_name, config.architecture, self._endpoint
        )
        self._freeze_tail = config.freeze_tail
        self._use_tail = config.use_tail

    def _build(self, inputs, is_training=False):
        pred = super(TruncatedBaseNetwork, self)._build(
            inputs, is_training=is_training)
        return self._get_endpoint(dict(pred['end_points']))

    def _get_endpoint(self, endpoints):
        for endpoints_key, endpoints_value in endpoints.items():
            if endpoints_key.endswith(self._scope_endpoint):
                return endpoints_value

    def _build_tail(self, inputs, is_training=False):
        if not self._use_tail:
            return inputs
        if self._architecture == 'resnet_v1_101':
            train_batch_norm = (
                is_training and self._config.get('train_batch_norm')
            )
            with self._enter_variable_scope():
                weight_decay = (self._config.get(
                    'arg_scope', {}).get('weight_decay', 0))
                with tf.variable_scope(self._architecture, reuse=True):
                    resnet_arg_scope = resnet_utils.resnet_arg_scope(
                        batch_norm_epsilon=1e-5, batch_norm_scale=True, weight_decay=weight_decay)
                    with slim.arg_scope(resnet_arg_scope):
                        with slim.arg_scope([slim.batch_norm], is_training=train_batch_norm):
                            blocks = [resnet_utils.Block('block4', resnet_v1.bottleneck, [{
                                'depth': 2048,
                                'depth_bottleneck': 512,
                                'stride': 1
                            }] * 3
                            )]
                            proposal_classifier_features = (
                                resnet_utils.stack_blocks_dense(inputs, blocks))
        else:
            proposal_classifier_features = inputs

        return proposal_classifier_features
