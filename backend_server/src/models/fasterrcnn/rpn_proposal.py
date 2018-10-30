import sonnet as snt
import tensorflow as tf
from src.utils.bbox_transform_tf import decode


class RPNProposal(snt.AbstractModule):
    def __init__(self, num_anchors, config, debug=False, name='proposal_layer'):
        super(RPNProposal, self).__init__(name=name)
        self._num_anchors = num_anchors
        self._pre_nms_top_n = config.pre_nms_top_n
        self._apply_nms = config.apply_nms
        self._post_nms_top_n = config.post_nms_top_n
        self._nms_threshold = float(config.nms_threshold)
        self._min_size = config.min_size
        self._filter_outside_anchors = config.filter_outside_anchors
        self._clip_after_nms = config.clip_after_nms
        self._min_prob_threshold = float(config.min_prob_threshold)
        self._debug = debug

    def _build(self, rpn_cls_prob, rpn_bbox_pred, all_anchors, im_shape):
        all_scores = rpn_cls_prob[:, 1]
        all_anchors = tf.reshape(all_scores, [-1])
        all_proposals = decode(all_anchors, rpn_bbox_pred)
