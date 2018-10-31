import tensorflow as tf


def decode(roi, deltas, variances=None):
    with tf.name_scope('BoundingBoxTransform/decode'):
        (roi_width, roi_height, roi_urx, roi_ury) = get_width_upright(roi)
        dx, dy, dw, dh = tf.split(deltas, 4, axis=1)
        if variances is None:
            variances = [1., 1.]
        pred_ur_x = dx * roi_width * variances[0] + roi_urx
        pred_ur_y = dy * roi_height * variances[0] + roi_ury
        pred_w = tf.exp(dw * variances[1]) * roi_width
        pred_h = tf.exp(dh * variances[1]) * roi_height

        bbox_x1 = pred_ur_x - .5 * pred_w
        bbox_y1 = pred_ur_y - .5 * pred_h
        bbox_x2 = pred_ur_x + .5 * pred_w - 1.
        bbox_y2 = pred_ur_y + .5 * pred_h - 1.
        bboxes = tf.concat([bbox_x1, bbox_y1, bbox_x2, bbox_y2], axis=1)
        return bboxes


def get_width_upright(bboxes):
    with tf.name_scope('BoundingBoxTransform/get_width_upright'):
        bboxes = tf.cast(bboxes, tf.float32)
        x1, y1, x2, y2 = tf.split(bboxes, 4, axis=1)
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        urx = x1 + .5 * width
        ury = y1 + .5 * height
        return width, height, urx, ury
