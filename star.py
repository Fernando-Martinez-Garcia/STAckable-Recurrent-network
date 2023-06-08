import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.nn.rnn_cell import LSTMStateTuple
from tensorflow.python.ops import random_ops
from tensorflow.python.keras import initializers


class STARCell(tf.compat.v1.nn.rnn_cell.BasicLSTMCell):
    def __init__(self, num_units, t_max=784,
                 **kwargs):
        '''
        t_max should be a float value corresponding to the longest possible
        time dependency in the input.
        '''
        self.num_units = num_units
        self.t_max = 784
        super(STARCell, self).__init__(num_units, **kwargs)

    def __call__(self, x, state, scope=None):
        """STAR cell."""
        with tf.compat.v1.variable_scope(scope or type(self).__name__):
            if self._state_is_tuple:
                h, _ = state
            else:
                h, _ = tf.split(value=state, num_or_size_splits=2, axis=1)

            all_inputs = tf.concat([x, h], 1)

            x_size = x.get_shape().as_list()[1]
            
            weights_OBS = tf.compat.v1.get_variable('W_xh_z',
                                   [x_size, 1 * self.num_units], initializer=initializers.get('orthogonal'))            
            W_xh = tf.compat.v1.get_variable('W_xh_K',
                                   [x_size, 1 * self.num_units], initializer=initializers.get('orthogonal'))
            W_hh = tf.compat.v1.get_variable('W_hh',
                                   [self.num_units, 1 * self.num_units], initializer=initializers.get('orthogonal'))
            

            if self.t_max is None:
                print('Zero initializer ')
                bias = tf.compat.v1.get_variable('bias', [2 * self.num_units],
                                       initializer=bias_initializer(2))
            else:
                print('Using chrono initializer ...')
                bias = tf.compat.v1.get_variable('bias', [2 * self.num_units],
                                       initializer=chrono_init(self.t_max,
                                                               2))

            weights_K = tf.concat([W_xh, W_hh], 0)
            
            bias_K = bias[self.num_units:,...]
            bias_OBS = bias[:self.num_units,...]
            

            f = tf.nn.bias_add(tf.matmul(all_inputs, weights_K), bias_K)
            j = tf.nn.bias_add(tf.matmul(x, weights_OBS), bias_OBS)

            beta = 1
            new_h = tf.sigmoid(f)*h + (1-tf.sigmoid(f-beta))*tf.tanh(j)
            new_h = tf.tanh(new_h)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_h, new_h)
            else:
                new_state = tf.concat([new_h, new_h], 1)
            return new_h, new_state


def chrono_init(t_max, num_gates):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        num_units = shape[0]//num_gates
        uni_vals = tf.math.log(random_ops.random_uniform([num_units], minval=1.0,
                                                    maxval=t_max, dtype=dtype,
                                                    seed=42))

        bias_j = tf.zeros(num_units)
        bias_f = uni_vals

        return tf.concat([bias_j, bias_f], 0)

    return _initializer


def bias_initializer(num_gates):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        p = np.zeros(shape)
        num_units = int(shape[0]//num_gates)
        # i, j, o, f
        # f:
        p[-num_units:] = np.ones(num_units)

        return tf.constant(p, dtype)

    return _initializer
