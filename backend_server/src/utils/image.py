import tensorflow as tf


def resize_image(image, bboxes=None, min_size=None, max_size=None):
    image_shape = tf.to_float(tf.shape(image))
    height = image_shape[0]
    width = image_shape[1]

    if min_size is not None:
        min_size = tf.to_float(min_size)
        min_dimension = tf.minimum(height, width)
        upscales_factor = tf.maximum(min_size/min_dimension, 1.)
    else:
        upscales_factor = tf.constant(1.)

    if max_size is not None:
        max_size = tf.to_float(max_size)
        max_dimension = tf.maximum(height, width)
        downscales_factor = tf.minimum(max_size/max_dimension, 1.)
    else:
        downscales_factor = tf.constant(1.)

    scale_factor = upscales_factor * downscales_factor

    new_height = height * scale_factor
    new_width = width * scale_factor
    image = tf.image.resize_images(image, tf.stack(tf.to_int32(
        [new_height, new_width])), method=tf.image.ResizeMethod.BILINEAR)

    return {
        'image': image,
        'scale_factor': scale_factor
    }


def flip_image(image, bboxes=None, left_right=True, up_down=False):
    image_shape = tf.shape(image)
    height = image_shape[0]
    width = image_shape[1]

    if bboxes is not None:
        # bboxes usually come from dataset as ints, but just in case we are
        # using flip for preprocessing, where bboxes usually are represented as
        # floats, we cast them.
        bboxes = tf.to_int32(bboxes)

    if left_right:
        image = tf.image.flip_left_right(image)
        if bboxes is not None:
            x_min, y_min, x_max, y_max, label = tf.unstack(bboxes, axis=1)
            new_x_min = width - x_max - 1
            new_y_min = y_min
            new_x_max = new_x_min + (x_max - x_min)
            new_y_max = y_max
            bboxes = tf.stack(
                [new_x_min, new_y_min, new_x_max, new_y_max, label], axis=1
            )

    if up_down:
        image = tf.image.flip_up_down(image)
        if bboxes is not None:
            x_min, y_min, x_max, y_max, label = tf.unstack(bboxes, axis=1)
            new_x_min = x_min
            new_y_min = height - y_max - 1
            new_x_max = x_max
            new_y_max = new_y_min + (y_max - y_min)
            bboxes = tf.stack(
                [new_x_min, new_y_min, new_x_max, new_y_max, label], axis=1
            )

    return_dict = {'image': image}
    if bboxes is not None:
        return_dict['bboxes'] = bboxes

    return return_dict
