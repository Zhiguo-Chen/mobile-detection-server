import tensorflow as tf


def get_initializer(initializer_config, seed=None):
    config = initializer_config.copy()
    config['seed'] = seed
    initializer = tf.random_normal_initializer(**config.pop('type'))
    return initializer
