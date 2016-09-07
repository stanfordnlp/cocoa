'''
RNN cell with attention over an input context.
'''

import tensorflow as tf
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell import _linear as linear

def add_attention_arguments(parser):
    parser.add_argument('--attn-scoring', default='linear', help='How to compute scores between hidden state and context {bilinear, linear}')
    parser.add_argument('--attn-output', default='project', help='How to combine rnn output and attention {concat, project}')

class AttnRNNCell(object):
    '''
    Abstract class for a rnn cell with attention mechanism.
    Derived class should follow signatures in RNNCell.
    '''

    def __init__(self, num_units, kg, rnn_type='lstm', scoring='linear', output='project', output_size=None, num_layers=1, activation=tanh):
        '''
        output_size: projected size of output + attention, used for prediction
        context_size: size of the context/attention vector
        '''
        self.rnn_cell = self._build_rnn_cell(rnn_type, num_units, num_layers)
        self._num_units = num_units
        self.kg = kg
        self._context_size = kg.context_size

        output_size = output_size or num_units
        if output == 'project':
            self._output_size = output_size
            self.output = self._output_project
        elif output == 'concat':
            self._output_size = self._num_units + self._context_size
            self.output = self._output_concat
        else:
            raise ValueError('Unknown output model')

        if scoring == 'linear':
            self.score_context = self._score_context_linear
        elif scoring == 'bilinear':
            self.score_context = self._score_context_bilinear
        else:
            raise ValueError('Unknown scoring model')

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

    def zero_state(self, batch_size, dtype):
        zero_rnn_state = self.rnn_cell.zero_state(batch_size, dtype)
        zero_h = tf.zeros([batch_size, self.rnn_cell.output_size])
        zero_attn, scores = self.compute_attention(zero_h, self.kg.context, self._num_units)
        return (zero_rnn_state, zero_attn)

    @property
    def output_size(self):
        return self._output_size

    def _score_context_linear(self, h, context):
        attn_size = self._num_units
        # Projection vector
        v = tf.get_variable('AttnProject', [attn_size, 1])
        def project_context(ctxt):
            a = tanh(linear([h, ctxt], attn_size, True))  # batch_size x attn_size
            return tf.squeeze(tf.matmul(a, v), [1])
        attns = tf.map_fn(project_context, tf.transpose(context, [1, 0, 2]))
        return attns

    def _score_context_bilinear(self, h, context):
        W = tf.get_variable('BilinearW', [self._num_units, self._context_size])
        def project_context(ctxt):
            return tf.reduce_sum(tf.mul(tf.matmul(h, W), ctxt), 1)
        attns = tf.map_fn(project_context, tf.transpose(context, [1, 0, 2]))
        return attns

    def _output_concat(self, output, attn):
        return tf.concat(1, [output, attn])

    def _output_project(self, output, attn):
        with tf.variable_scope("AttnOutputProjection"):
            new_output = tanh(linear([output, attn], self._output_size, True))
        return new_output

    def compute_attention(self, h, context, attn_size):
        '''
        context: batch_size x context_len x context_size
        attn_size: vector size used for scoring each context
        '''
        with tf.variable_scope('Attention'):
            attn_scores = self.score_context(h, context)
            attn_scores = tf.transpose(attn_scores) # batch_size x context_len
            attns = tf.nn.softmax(attn_scores)
            # Compute attention weighted context
            attns = tf.expand_dims(attns, 2)
            weighted_context = tf.reduce_sum(tf.mul(attns, context), 1)  # batch_size x context_size
            return weighted_context, attn_scores

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            inputs, entities = inputs
            prev_rnn_state, prev_attn = state
            # RNN step
            new_inputs = tf.concat(1, [inputs, prev_attn])
            output, rnn_state = self.rnn_cell(new_inputs, prev_rnn_state)
            # Update graph
            update_op = self.kg.update_utterance(entities, output)
            # Compute attention
            # context: batch_size x context_len x context_size
            # NOTE: kg.context assumes batch_size=1
            # TODO: in tensorflow self.kg.context is computed in every RNN step. Need to use partial_run to manually cache the intermediate result.
            with tf.control_dependencies([update_op]):
                attn, attn_scores = self.compute_attention(output, self.kg.context, self._num_units)
                # Output
                new_output = self.output(output, attn)
                #attn_scores = tf.sparse_to_dense(self.kg.entity_indices, tf.constant([attn_scores.get_shape()[0], self.kg.total_num_entities]), attn_scores)
            return (new_output, attn_scores), (rnn_state, attn)

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
        cell = AttnRNNCell(5, 5, 'lstm') # num_units=5, context_size=5
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




