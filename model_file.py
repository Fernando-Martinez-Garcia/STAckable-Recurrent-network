import tensorflow as tf
from tf_ops import linear
from star import STARCell
from bn_star import BNSTAR_cell

tf.compat.v1.disable_eager_execution()


def rnn(x, h_dim, y_dim, keep_prob, sequence_lengths,
        training, output_format, cell_type='star',
        t_max=784):
    '''
    Inputs:
    x - The input data.
        Tensor shape (batch_size, max_sequence_length, n_features)
    h_dim - A list with the number of neurons in each hidden layer
    keep_prob - The percentage of weights to keep in each iteration
                 (1-drop_prob)

    Returns:
    The non-softmaxed output of the RNN.
    '''

    def single_cell(dim, output_projection=None, training=True):
        if cell_type == 'rnn':
            print('Using the standard RNN cell')
            cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(dim)        
        elif cell_type == 'lstm':
            cell = tf.compat.v1.nn.rnn_cell.LSTMCell(dim)  
            print('Using LSTM cell')
        elif cell_type == 'star':
            cell = STARCell(dim, t_max=t_max)
            print('Using STAR cell')
        elif cell_type == 'bn-star':
            cell = BNSTAR_cell(dim, t_max=t_max, training=training)
            print('Using BN-STAR cell')


        drop_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
            cell,
            input_keep_prob=1,
            output_keep_prob=keep_prob)
        return drop_cell

    if len(h_dim) > 1:
        # Multilayer RNN        
        cells = [single_cell(dim, training=training) for dim in h_dim]
        
        print('Num cells: ' + str(len(cells)))
        cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells)
    else:
        cell = single_cell(h_dim[0], training=training)

    if output_format == 'last':
        out , final_state = tf.compat.v1.nn.dynamic_rnn(
            cell, x, sequence_length=sequence_lengths,
            dtype=tf.float32)


        # self.final_state is a tuple with
        # (state.c, state.h)
        if len(h_dim) > 1:
            # If we have a multi-layer rnn, get the top layer state
            out = final_state[-1]
            out = out[1]
        else:
            out = final_state
            out = out[1]
        
        proj_out = linear(out, y_dim, scope='output_mapping')
    
    elif output_format == 'all':
        out, _ = tf.compat.v1.nn.dynamic_rnn(
            cell, x, sequence_length=sequence_lengths,
            dtype=tf.float32)
        flat_out = tf.reshape(out, (-1, out.get_shape()[-1]))
        proj_out = linear(flat_out, y_dim, scope='output_mapping')
        proj_out = tf.reshape(proj_out,
                              (tf.shape(out)[0], tf.shape(out)[1], y_dim))

    return proj_out


class RNN_Model(object):

    def __init__(self, n_features, n_classes, h_dim, max_sequence_length,
                 is_test=False, max_gradient_norm=None, opt_method='adam',
                 learning_rate=0.001, weight_decay=0,
                 cell_type='star', chrono=False, mse=False,
                 ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.h_dim = h_dim
        self.max_sequence_length = max_sequence_length
        self.opt_method = opt_method
        #self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.is_test = is_test
        self.weight_decay = weight_decay
        self.cell_type = cell_type
        self.chrono = chrono
        self.mse = mse

    def build_inputs(self):
        self.keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
        self.x = tf.compat.v1.placeholder(tf.float32, [None, None, self.n_features],
                                name='x')
        if self.output_seq:
            self.y = tf.compat.v1.placeholder(tf.float32, [None, None, self.n_classes],
                                    name='y')
        else:
            self.y = tf.compat.v1.placeholder(tf.float32, [None, self.n_classes],
                                    name='y')
        self.seq_lens = tf.compat.v1.placeholder(tf.int32, [None],
                                       name="sequence_lengths")
        self.training = tf.compat.v1.placeholder(tf.bool)

        self.learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[])
        
        
    def build_loss(self, outputs):
        if self.mse:
            mean_squared_error = tf.losses.mean_squared_error(
                labels=self.y, predictions=outputs)
            self.loss_nowd = tf.reduce_mean(mean_squared_error)
            tf.summary.scalar('mean_squared_error',
                              tf.reduce_mean(mean_squared_error))
        else:
            if self.output_seq:
                flat_out = tf.reshape(outputs, (-1, tf.shape(outputs)[-1]))
                flat_y = tf.reshape(self.y, (-1, tf.shape(self.y)[-1]))
                sample_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=flat_y, logits=flat_out)

            else:
                sample_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.y, logits=outputs)

            tf.summary.scalar('cross_entropy',
                              tf.reduce_mean(sample_cross_entropy))
            self.loss_nowd = tf.reduce_mean(sample_cross_entropy)

        weight_decay = self.weight_decay*tf.add_n([tf.nn.l2_loss(v) for v in
                                                   tf.compat.v1.trainable_variables()
                                                   if 'bias' not in v.name])
        tf.summary.scalar('weight_decay', weight_decay)
        self.loss = self.loss_nowd + weight_decay

    def build_optimizer(self):
        if self.opt_method == 'adam':
            print('Optimizing with Adam')
            opt = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        elif self.opt_method == 'rms':
            print('Optimizing with RMSProp')
            opt = tf.compat.v1.train.RMSPropOptimizer(self.learning_rate)
        elif self.opt_method == 'momentum':
            print('Optimizing with Nesterov momentum SGD')
            opt = tf.train.MomentumOptimizer(self.learning_rate,
                                             momentum=0.9,
                                             use_nesterov=True)

        params = tf.compat.v1.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         self.max_gradient_norm)
        tf.summary.scalar('gradients_norm', norm)

        self.train_opt = opt.apply_gradients(zip(clipped_gradients, params))

    def build(self, output_format='last'):
        print("Building model ...")
        self.output_seq = False
        if output_format == 'all':
            self.output_seq = True
        self.build_inputs()

        t_max = None
        if self.chrono:
            t_max = self.max_sequence_length

        outputs = rnn(self.x, self.h_dim, self.n_classes, self.keep_prob,
                      sequence_lengths=self.seq_lens,
                      training=self.training, output_format=output_format,
                      cell_type=self.cell_type, t_max=t_max,
                      )

        self.build_loss(outputs)

        self.output_probs = tf.nn.softmax(outputs)
        if self.output_seq:
            self.output_probs = tf.reshape(
                self.output_probs, (-1, tf.shape(self.output_probs)[-1]))

        if not self.is_test:
            print("Adding training operations")
            self.build_optimizer()

        self.summary_op = tf.compat.v1.summary.merge_all()
