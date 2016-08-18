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
from tensorflow.python.ops import variable_scope as vs

class AttnRNNCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, context_size, input_size=None, output_size=None, num_layers=1, activation=tanh):
        '''
        input_size: projected size of input + attention, feed to rnn
        output_size: projected size of output + attention, used for prediction
        context_size: size of the context/attention vector
        '''
        self.rnn_cell = self._build_rnn_cell(num_units, num_layers)
        self._num_units = num_units
        self._input_size = input_size or num_units
        self._output_size = output_size or num_units
        self._context_size = context_size

    def compute_attention(self, h, context, attn_size):
        '''
        context: batch_size x context_len x context_size
        attn_size: vector size used for scoring each context
        '''
        with vs.variable_scope('Attention'):
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

class AttnBasicRNNCell(AttnRNNCell):
    def _build_rnn_cell(self, num_units, num_layers=1):
        cell = tf.nn.rnn_cell.BasicRNNCell(num_units)
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
        return cell

    @property
    def state_size(self):
        return (self._num_units, self._context_size)

    @property
    def output_size(self):
        return self._output_size

    def zero_state(self, init_context, batch_size, dtype):
        zero_h = self.rnn_cell.zero_state(batch_size, dtype)
        zero_attn = self.compute_attention(zero_h, init_context, self._num_units)
        return (zero_h, zero_attn)

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or type(self).__name__): # "AttnBasicRNNCell"
            # context: batch_size x context_len x context_size
            inputs, context = inputs
            prev_h, prev_attn = state
            # RNN step. h: batch_size x rnn_size
            with vs.variable_scope("AttnInputProjection"):
                new_inputs = linear([inputs, prev_attn], self._input_size, True)
            output, h = self.rnn_cell(new_inputs, prev_h)
            # Compute attention
            attn = self.compute_attention(h, context, self._num_units)
            # Output
            with vs.variable_scope("AttnOutputProjection"):
                new_output = linear([output, attn], self._output_size, True)
            return new_output, (h, attn)

# test
if __name__ == '__main__':
    # Test single cell step
    with tf.variable_scope('test1'):
        cell = AttnBasicRNNCell(10, 8, 12)
        inputs = tf.ones([1,5], dtype=tf.float32)
        context = tf.random_uniform([1,2,3])
        init_state = cell.zero_state(context, 1, tf.float32)
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            output, state = cell((inputs, context), init_state)
            vs.get_variable_scope().reuse_variables()
            output, state = cell((inputs, context), state)

    # Test dynamic rnn
    T = 10
    # NOTE: current TF does not support rand>3 matrix with time_major=False!
    with tf.variable_scope('test2'):
        inputs = tf.ones([T,1,5], dtype=tf.float32)
        context = tf.random_uniform([T,1,2,3])
        outputs, states = tf.nn.dynamic_rnn(cell, (inputs, context), initial_state=init_state, time_major=True)
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            outputs, states = sess.run([outputs, states])
            print len(outputs), outputs[0].shape, outputs[1].shape
            print len(states), states[0].shape, states[1].shape




