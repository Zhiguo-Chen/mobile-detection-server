import sonnet as snt
import tensorflow as tf
from src.utils.bbox_transform_tf import decode, change_order


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
        all_scores = tf.reshape(all_scores, [-1])
        all_proposals = decode(all_anchors, rpn_bbox_pred)
        min_prob_filter = tf.greater_equal(
            all_scores, self._min_prob_threshold)
        (x_min, y_min, x_max, y_max) = tf.unstack(all_proposals, axis=1)
        zero_area_filter = tf.greater(tf.maximum(
            x_max - x_min, 0.0) * tf.maximum(y_max - y_min, 0.0), 0.0)
        proposal_filter = tf.logical_and(zero_area_filter, min_prob_filter)
        all_proposals_total = tf.shape(all_scores)[0]
        unsorted_scores = tf.boolean_mask(
            all_scores, proposal_filter, name='filtered_scores')
        unsorted_proposals = tf.boolean_mask(
            all_proposals, proposal_filter, name='filtered_proposals')
        filtered_proposals_total = tf.shape(unsorted_scores)[0]
        tf.summary.scalar(
            'valid_proposals_ratio',
            (
                tf.cast(filtered_proposals_total, tf.float32) /
                tf.cast(all_proposals_total, tf.float32)
            ), ['rpn'])

        tf.summary.scalar(
            'invalid_proposals',
            all_proposals_total - filtered_proposals_total, ['rpn'])

        k = tf.minimum(self._pre_nms_top_n, tf.shape(unsorted_scores)[0])
        top_k = tf.nn.top_k(unsorted_scores, k=k)
        sorted_top_proposals = tf.gather(unsorted_proposals, top_k.indices)
        sorted_top_scores = top_k.values

        if self._apply_nms:
            with tf.name_scope('nms'):
                proposals_tf_order = change_order(sorted_top_proposals)
                selected_indices = tf.image.non_max_suppression(proposals_tf_order, tf.reshape(
                    sorted_top_scores, [-1]), self._post_nms_top_n, iou_threshold=self._nms_threshold)
                nms_proposals_tf_order = tf.gather(
                    proposals_tf_order, selected_indices, name='gather_nms_proposals')
                proposals = change_order(nms_proposals_tf_order)
                scores = tf.gather(
                    sorted_top_proposals, selected_indices, name='gather_nms_proposals_scores')

        pred = {
            'proposals': proposals,
            'scores': scores,
        }

        return pred
