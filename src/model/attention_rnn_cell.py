'''
Reference: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/seq2seq.py

First, we run the cell on a combination of the input and previous
attention masks:
    cell_output, new_state = cell(linear(input, prev_attn), prev_state).
Then, we calculate new attention masks:
    new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
and then we calculate the output:
    output = linear(cell_output, new_attn).
'''

import tensorflow as tf
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell import _linear as linear

class AttnRNNCell(object):
    '''
    Abstract class for a rnn cell with attention mechanism.
    Derived class should follow signatures in RNNCell.
    '''

    def __init__(self, num_units, rnn_type='lstm', input_size=None, output_size=None, num_layers=1, activation=tanh):
        '''
        input_size: projected size of input + attention, feed to rnn
        output_size: projected size of output + attention, used for prediction
        context_size: size of the context/attention vector
        '''
        self.rnn_cell = self._build_rnn_cell(rnn_type, num_units, num_layers)
        self._num_units = num_units
        self._input_size = input_size or num_units
        self._output_size = output_size or num_units

    def _build_rnn_cell(self, rnn_type, rnn_size, num_layers):
        recurrent_cell = {'rnn': tf.nn.rnn_cell.BasicRNNCell,
                          'gru': tf.nn.rnn_cell.GRUCell,
                          'lstm': tf.nn.rnn_cell.LSTMCell,
                         }

        cell = None
        if rnn_type == 'lstm':
            cell = recurrent_cell[rnn_type](rnn_size, state_is_tuple=True)
        else:
            cell = recurrent_cell[rnn_type](rnn_size)
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
        return cell

    def zero_state(self, init_context, batch_size, dtype):
        zero_rnn_state = self.rnn_cell.zero_state(batch_size, dtype)
        zero_h = tf.zeros([batch_size, self.rnn_cell.output_size])
        zero_attn = self.compute_attention(zero_h, init_context, self._num_units)
        return (zero_rnn_state, zero_attn, init_context)

    @property
    def output_size(self):
        return self._output_size

    def compute_attention(self, h, context, attn_size):
        '''
        context: batch_size x context_len x context_size
        attn_size: vector size used for scoring each context
        '''
        return tf.reduce_mean(context, 1)
        with tf.variable_scope('Attention'):
            # Projection vector
            v = tf.get_variable('AttnProject', [attn_size, 1])
            def project_context(ctxt):
                a = tanh(linear([h, ctxt], attn_size, True))  # batch_size x attn_size
                return tf.matmul(a, v)
            attns = tf.map_fn(project_context, tf.transpose(context, [1, 0, 2]))
            attns = tf.transpose(tf.squeeze(attns, [2])) # batch_size x context_len
            attns = tf.nn.softmax(attns)
            # Compute attention weighted context
            attns = tf.expand_dims(attns, 2)
            weighted_context = tf.reduce_sum(tf.mul(attns, context), 1)  # batch_size x context_size
            return weighted_context

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # context: batch_size x context_len x context_size
            prev_rnn_state, prev_attn, context = state
            # RNN step
            #with tf.variable_scope("AttnInputProjection"):
            #    new_inputs = linear([inputs, prev_attn], self._input_size, True)
            new_inputs = tf.concat(1, [inputs, prev_attn])
            output, rnn_state = self.rnn_cell(new_inputs, prev_rnn_state)
            # Compute attention
            attn = self.compute_attention(output, context, self._num_units)
            # Output
            with tf.variable_scope("AttnOutputProjection"):
                new_output = tanh(linear([output, attn], self._output_size, True))
            return new_output, (rnn_state, attn, context)
            #return output, (rnn_state, attn, context)

# test
if __name__ == '__main__':
    # Test single cell step
    with tf.variable_scope('test1'):
        cell = AttnRNNCell(10, 'lstm', 8, 12)
        inputs = tf.ones([1,5], dtype=tf.float32)
        context = tf.random_uniform([1,2,3])
        init_state = cell.zero_state(context, 1, tf.float32)
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            output, state = cell(inputs, init_state)
            tf.get_variable_scope().reuse_variables()
            output, state = cell(inputs, state)

    # Test dynamic rnn
    T = 10
    # NOTE: current TF does not support rand>3 matrix with time_major=False!
    with tf.variable_scope('test2'):
        cell = AttnRNNCell(5, 'lstm') # num_units=5, context_size=5
        context = tf.random_uniform([1,4,5])
        init_state = cell.zero_state(context, 1, tf.float32)
        inputs = tf.ones([T,1,5], dtype=tf.float32)
        outputs, states = tf.scan(lambda a, x: cell(x, a[1]), elems=inputs, initializer=(tf.zeros([1, 5]), init_state), parallel_iterations=1)

        w = tf.get_variable('output_w', [5, 2])
        b = tf.get_variable('output_b', [2])
        outputs = tf.map_fn(lambda x: tf.matmul(x, w) + b, outputs)
        target = tf.ones([10,1], dtype=tf.int32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(outputs, target)
        tvars = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        grads_and_vars = optimizer.compute_gradients(loss, tvars)
        train_op = optimizer.apply_gradients(grads_and_vars)
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            outputs, states = sess.run([outputs, states])
            print len(outputs), outputs[0].shape, outputs[1].shape
            sess.run(train_op)
            sess.run(train_op)
            print 'run gradient'




