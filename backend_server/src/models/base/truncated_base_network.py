import tensorflow as tf
from .base_network import BaseNetwork


class TruncatedBaseNetwork(BaseNetwork):
    def __init__(self, config, name='truncated_base_network', **kwargs):
        super(TruncatedBaseNetwork, self).__init__(config, name=name)

        def _build(self, inputs, is_training=False):
            pred = super(TruncatedBaseNetwork, self)._build(
                inputs, is_training=is_training)
