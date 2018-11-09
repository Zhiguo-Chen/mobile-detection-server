import sonnet as snt
import tensorflow as tf

from src.utils.bbox_transform_tf import decode, clip_boxes, change_order


class RCNNProposal(snt.AbstractModule):
    def __init__(self, num_classes, config, variances=None, name='rcnn_proposal'):
        super(RCNNProposal, self).__init__(name=name)
        self._num_classes = num_classes
        self._variances = variances
        self._class_max_detection = config.class_max_detections
        self._class_nms_threshold = float(config.class_nms_threshold)
        self._total_max_detections = config.total_max_detections
        self._min_prob_threshold = config.min_prob_threshold or 0.0

    def _build(self, proposals, bbox_pred, cls_prob, im_shape):
        selected_boxes = []
        selected_probs = []
        selected_labels = []
        for class_id in range(self._num_classes):
            class_prob = cls_prob[:, class_id + 1]
            class_bboxes = bbox_pred[:, (class_id * 4):((class_id + 1) * 4)]
            raw_class_objects = decode(
                proposals, class_bboxes, variances=self._variances)
            class_objects = clip_boxes(raw_class_objects, im_shape)
            prob_filter = tf.greater_equal(
                class_prob, self._min_prob_threshold)
            (x_min, y_min, x_max, y_max) = tf.unstack(class_objects, axis=1)
            area_filter = tf.greater(tf.maximum(
                x_max - x_min, 0.0) * tf.maximum(y_max - y_min, 0.0), 0.0)
            object_filter = tf.logical_and(area_filter, prob_filter)
            class_objects = tf.boolean_mask(class_objects, object_filter)
            class_prob = tf.boolean_mask(class_prob, object_filter)
            class_objects_tf = change_order(class_objects)
            class_selected_idx = tf.image.non_max_suppression(
                class_objects_tf, class_prob, self._class_max_detection, iou_threshold=self._class_nms_threshold)
            class_objects_tf = tf.gather(class_objects_tf, class_selected_idx)
            class_prob = tf.gather(class_prob, class_selected_idx)
            class_objects = change_order(class_objects_tf)
            selected_boxes.append(class_objects)
            selected_probs.append(class_prob)
            selected_labels.append(
                tf.tile([class_id], [tf.shape(class_selected_idx)[0]]))
        objects = tf.concat(selected_boxes, axis=0)
        proposal_label = tf.concat(selected_labels, axis=0)
        proposal_label_prob = tf.concat(selected_probs, axis=0)
        tf.summary.histogram('proposal_cls_scores',
                             proposal_label_prob, ['rcnn'])
        k = tf.minimum(self._total_max_detections,
                       tf.shape(proposal_label_prob)[0])
        top_k = tf.nn.top_k(proposal_label_prob, k=k)
        top_k_proposal_label_prob = top_k.values
        top_k_objects = tf.gather(objects, top_k.indices)
        top_k_proposal_label = tf.gather(proposal_label, top_k.indices)
        return {
            'objects': top_k_objects,
            'proposal_label': top_k_proposal_label,
            'proposal_label_prob': top_k_proposal_label_prob,
            'selected_boxes': selected_boxes,
            'selected_probs': selected_probs,
            'selected_labels': selected_labels,
        }
