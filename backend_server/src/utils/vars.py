import tensorflow as tf


VAR_LOG_LEVELS = {
    'full': ['variable_summaries_full'],
    'reduced': ['variable_summaries_reduced', 'variable_summaries_full']
}


def get_initializer(initializer_config, seed=None):
    config = initializer_config.copy()
    config['seed'] = seed
    initializer = tf.random_normal_initializer(**config.pop('type'))
    return initializer


def get_activation_function(activation_function):
    try:
        return getattr(tf.nn, activation_function)
    except AttributeError:
        raise ValueError(
            'Invalid activation function "{}"'.format(activation_function)
        )


def variable_summaries(var, name, collection_key):
    if collection_key not in VAR_LOG_LEVELS.keys():
        raise ValueError('"{}" not in `VAR_LOG_LEVELS`'.format(collection_key))
    collections = VAR_LOG_LEVELS[collection_key]
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean, collections)
        num_params = tf.reduce_prod(tf.shape(var))
        tf.summary.scalar('num_params', num_params, collections)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev, collections)
        tf.summary.scalar('max', tf.reduce_max(var), collections)
        tf.summary.scalar('min', tf.reduce_min(var), collections)
        tf.summary.histogram('histogram', var, collections)
        tf.summary.scalar('sparsity', tf.nn.zero_fraction(var), collections)


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
