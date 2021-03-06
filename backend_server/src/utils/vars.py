import tensorflow as tf


VAR_LOG_LEVELS = {
    'full': ['variable_summaries_full'],
    'reduced': ['variable_summaries_reduced', 'variable_summaries_full']
}

VALID_INITIALIZERS = {
    'truncated_normal_initializer': tf.truncated_normal_initializer,
    'variance_scaling_initializer': (
        tf.contrib.layers.variance_scaling_initializer
    ),
    'random_normal_initializer': tf.random_normal_initializer,
    'xavier_initializer': tf.contrib.layers.xavier_initializer,
}


def get_initializer(initializer_config, seed=None):
    """Get variable initializer.

    Args:
        - initializer_config: Configuration for initializer.

    Returns:
        initializer: Instantiated variable initializer.
    """

    if 'type' not in initializer_config:
        raise ValueError('Initializer missing type.')

    if initializer_config.type not in VALID_INITIALIZERS:
        raise ValueError('Initializer "{}" is not valid.'.format(
            initializer_config.type))

    config = initializer_config.copy()
    initializer = VALID_INITIALIZERS[config.pop('type')]
    config['seed'] = seed

    return initializer(**config)


def get_activation_function(activation_function):
    if not activation_function:
        return lambda a: a
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
