import sonnet as snt


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
