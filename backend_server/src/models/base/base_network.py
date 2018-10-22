import sonnet as snt
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1, resnet_v2

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


class BaseNetwork(snt.AbstractModule):
    def __init__(self, config, name='base_network'):
        super(BaseNetwork, self).__init__(name=name)
        self._architecture = config.get('architecture')
        self._config = config
        self.pretrained_weights_scope = None

    @property
    def arg_scope(self):
        arg_scope_kwargs = self._config.get('arg_scope', {})

        return resnet_v2.resnet_utils.resnet_arg_scope(**arg_scope_kwargs)

    def _build(self, inputs, is_training=False):
        inputs = self._subtract_channels(inputs)

    def _subtract_channels(self, inputs, mean=[_R_MEAN, _G_MEAN, _B_MEAN]):
        return inputs - mean

    def network(self, is_training=False):
        output_stride = self._config.get('output_stride')
