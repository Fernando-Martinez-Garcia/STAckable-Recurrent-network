import tensorflow as tf


def create_sess():
    sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    return tf.compat.v1.Session(config=sess_config)


def create_scalar_summaries(tags, values):
    '''
    Input:
    tags - the tag names to use in the summary
    values - the values to log

    Returns:
    A tensorflow summary to be used in wrtier.add_summary(summary, steps)
    '''
    summary_value_list = []
    for i in range(len(tags)):
        summary_value_list.append(tf.compat.v1.Summary.Value(tag=tags[i],
                                                   simple_value=values[i]))
    return tf.compat.v1.Summary(value=summary_value_list)


def linear(input_data, output_dim, scope=None, stddev=1.0, init_func=None):
    if init_func == 'norm':
        initializer = tf.random_normal_initializer(stddev=stddev)
    elif init_func is None:
        initializer = None
    const = tf.constant_initializer(0.0)
    with tf.compat.v1.variable_scope(scope or 'linear'):
        w = tf.compat.v1.get_variable(
            'weights', [input_data.get_shape()[-1], output_dim],
            initializer=initializer)
        b = tf.compat.v1.get_variable('bias', [output_dim], initializer=const)
        return tf.matmul(input_data, w) + b
