import tensorflow as tf
from .base_network import BaseNetwork

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

    def _build(self, inputs, is_training=False):
        pred = super(TruncatedBaseNetwork, self)._build(
            inputs, is_training=is_training)
        return self._get_endpoint(dict(pred['end_points']))

    def _get_endpoint(self, endpoints):
        for endpoints_key, endpoints_value in endpoints.items():
            if endpoints_key.endswith(self._scope_endpoint):
                return endpoints_value
