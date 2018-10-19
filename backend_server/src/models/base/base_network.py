import sonnet as snt


class BaseNetwork(snt.AbstractModule):
    def __init__(self, config, name='base_network'):
        super(BaseNetwork, self).__init__(name=name)

    def _build(self, input, is_training=False):
        pass
