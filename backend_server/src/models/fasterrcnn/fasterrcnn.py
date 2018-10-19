import tensorflow as tf
import sonnet as snt
from src.utils.anchors import generate_anchors_reference
import numpy as np


class FasterRCNN(snt.AbstractModule):
    def __init__(self, config, name='fasterrcnn'):
        super(FasterRCNN, self).__init__(name=name)
        self._congig = config
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

    def _build(self, image, gt_box=None, is_training=False):
        pass
