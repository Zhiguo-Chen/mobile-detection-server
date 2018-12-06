import sonnet as snt
import tensorflow as tf


class ROIPoolingLayer(snt.AbstractModule):
    def __init__(self, config, debug=False, name='roi_pooling'):
        super(ROIPoolingLayer, self).__init__(name=name)
        self._pooling_mode = config.pooling_mode.lower()
        self._pooled_width = config.pooled_width
        self._pooled_height = config.pooled_height
        self._pooled_padding = config.padding
        self._debug = debug

    def _build(self, roi_proposals, conv_feature_map, im_shape):
        return self._roi_crop(roi_proposals, conv_feature_map, im_shape)

    def _roi_crop(self, roi_proposals, conv_feature_map, im_shape):
        bboxes = self._get_bboxes(roi_proposals, im_shape)
        bboxes_shape = tf.shape(bboxes)
        batch_ids = tf.zeros((bboxes_shape[0], ), dtype=tf.int32)
        crops = tf.image.crop_and_resize(conv_feature_map, bboxes, batch_ids, [
                                         self._pooled_width * 2, self._pooled_height * 2], name='crops')
        prediction_dict = {'roi_pool': tf.nn.max_pool(
            crops, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self._pooled_padding)}
        return prediction_dict

    def _get_bboxes(self, roi_proposals, im_shape):
        with tf.name_scope('get_bboxes'):
            im_shape = tf.cast(im_shape, tf.float32)
            x1, y1, x2, y2 = tf.unstack(roi_proposals, axis=1)
            x1 = x1 / im_shape[1]
            y1 = y1 / im_shape[0]
            x2 = x2 / im_shape[1]
            y2 = y2 / im_shape[0]
            bboxes = tf.stack([y1, x1, y2, x2], axis=1)
            return bboxes
