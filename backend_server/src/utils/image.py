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
