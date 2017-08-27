'''
RNN cell with attention over an input context.
'''

import tensorflow as tf
from cocoa.model.util import linear, batch_linear, batch_embedding_lookup, EPS
from cocoa.model.rnn_cell import build_rnn_cell

def add_attention_arguments(parser):
    parser.add_argument('--attn-scoring', default='linear', help='How to compute scores between hidden state and context {bilinear, linear}')
    parser.add_argument('--attn-output', default='project', help='How to combine rnn output and attention {concat, project}')
    parser.add_argument('--no-checklist', default=False, action='store_true', help='Whether to include checklist at each RNN step')

class AttnRNNCell(object):
    '''
    RNN cell with attention mechanism over an input context.
    '''

    def __init__(self, rnn_size, context_size, rnn_type='lstm', keep_prob=1, scoring='linear', output='project', num_layers=1, checklist=True):
        self.rnn_cell = build_rnn_cell(rnn_type, rnn_size, num_layers, keep_prob)
        self.rnn_size = rnn_size
        self.context_size = context_size
        self.checklist = checklist

        self.scorer = scoring
        self.output_combiner = output

        if self.output_combiner == 'project':
            self.output_size = self.rnn_size
        elif self.output_combiner == 'concat':
            self.output_size = self.rnn_size + self.context_size
        else:
            raise ValueError('Unknown output model')

    def init_state(self, rnn_state, rnn_output, context, checklist):
        attn, scores = self.compute_attention(rnn_output, context, checklist)
        return (rnn_state, attn, context)

    def zero_state(self, batch_size, init_context, dtype=tf.float32):
        zero_rnn_state = self.rnn_cell.zero_state(batch_size, dtype)
        zero_h = tf.zeros([batch_size, self.rnn_cell.output_size], dtype=dtype)
        zero_checklist = tf.zeros_like(init_context)[:, :, 0]
        return self.init_state(zero_rnn_state, zero_h, init_context, zero_checklist)

    def score_context(self, h, context, checklist):
        # Repeat h for each cell in context
        context_len = tf.shape(context)[1]
        h = tf.tile(tf.expand_dims(h, 1), [1, context_len, 1])  # (batch_size, context_len, rnn_size)

        if self.scorer == 'linear':
            return self._score_context_linear(h, context, checklist)
        elif self.scorer == 'bilinear':
            return self._score_context_bilinear(h, context)
        else:
            raise ValueError('Unknown scoring model')

    def _score_context_linear(self, h, context, checklist):
        '''
        Concatenate state h and context, combine them to a vector, then project to a scalar.
        h: (batch_size, context_len, rnn_size)
        context: (batch_size, context_len, context_size)
        checklist: (batch_size, context_len, 1)
        Return context_scores (batch_size, context_len)
        '''
        attn_size = self.rnn_size
        with tf.variable_scope('ScoreContextLinear'):
            with tf.variable_scope('Combine'):
                if self.checklist:
                    feature = [h, context, checklist]
                else:
                    feature = [h, context]
                attns = activation(batch_linear(feature, attn_size, False))  # (batch_size, context_len, attn_size)
            with tf.variable_scope('Project'):
                attns = tf.squeeze(batch_linear(attns, 1, False), [2])  # (batch_size, context_len)
                #if self.checklist:
                #    weight = tf.get_variable('cl_weight', [])
                #    attns = attns - tf.scalar_mul(weight, tf.squeeze(checklist, [2]))
        return attns

    def _score_context_bilinear(self, h, context):
        '''
        Project h to context_size then do dot-product with context.
        h: (batch_size, context_len, rnn_size)
        context: (batch_size, context_len, context_size)
        Return context_scores (batch_size, context_len)
        '''
        context_size = context.get_shape().as_list()[-1]
        with tf.variable_scope('ScoreContextBilinear'):
            h = batch_linear(h, context_size, False)  # (batch_size, context_len, context_size)
            attns = tf.reduce_sum(tf.mul(h, context), 2)  # (batch_size, context_len)
        return attns

    def output_with_attention(self, output, attn):
        '''
        Combine rnn output and the attention vector to generate the final output.
        '''
        if self.output_combiner == 'project':
            return self._output_project(output, attn, self.output_size)
        elif self.output_combiner == 'concat':
            return self._output_concat(output, attn)
        else:
            raise ValueError('Unknown output model')

    def _output_concat(self, output, attn):
        return tf.concat(1, [output, attn])

    def _output_project(self, output, attn, project_size):
        with tf.variable_scope("AttnOutputProjection"):
            new_output = activation(linear([output, attn], project_size, False))
        return new_output

    def compute_attention(self, h, context, checklist):
        '''
        context_mask filteres padded context cell, i.e., their attn_score is -inf.
        context: (batch_size, context_len, context_size)
        context_mask: (batch_size, context_len)
        checklist: (batch_size, context_len)
        '''
        with tf.variable_scope('Attention'):
            context, context_mask = context
            checklist = tf.expand_dims(checklist, 2)  # (batch_size, context_len, 1)
            with tf.variable_scope("ScoreAttention"):
                attn_scores = self.score_context(h, context, checklist)  # (batch_size, context_len)
            attns = tf.nn.softmax(attn_scores)
            zero_attns = tf.zeros_like(attns)
            attns = tf.where(context_mask, attns, zero_attns)
            # Compute attention weighted context
            attns = tf.expand_dims(attns, 2)
            weighted_context = tf.reduce_sum(tf.mul(attns, context), 1)  # (batch_size, context_size)
            # Setting it to -inf seems to cause learning problems
            neginf = -10. * tf.ones_like(context_mask, dtype=tf.float32)
            masked_attn_scores = tf.where(context_mask, attn_scores, neginf)
            return weighted_context, masked_attn_scores

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            prev_rnn_state, prev_attn, prev_context = state
            inputs, checklist = inputs
            # RNN step
            new_inputs = tf.concat(1, [inputs, prev_attn])
            output, rnn_state = self.rnn_cell(new_inputs, prev_rnn_state)
            # No update in context inside an utterance
            attn, attn_scores = self.compute_attention(output, prev_context, checklist)
            # Output
            new_output = self.output_with_attention(output, attn)
            return (new_output, attn_scores), (rnn_state, attn, prev_context)

class PreselectAttnRNNCell(AttnRNNCell):
    '''
    Attention RNN cell that pre-selects a set of items from the context.
    '''
    def select(self, init_output, context):
        context_len = tf.shape(context)[1]
        init_state = tf.tile(tf.expand_dims(init_output, 1), [1, context_len, 1])  # (batch_size, context_len, rnn_size)
        with tf.variable_scope('SelectEntity'):
            selection = batch_linear(tf.concat(2, [init_state, context]), 1, True)  # (batch_size, context_len, 1)
            selection_scores = tf.squeeze(selection, [2])
            selection = tf.sigmoid(selection)
            selected_context = tf.reduce_sum(tf.mul(selection, context), 1)  # (batch_size, context_size)
            # Normalize
            selected_context = tf.div(selected_context, (tf.reduce_sum(selection, 1) + EPS))
        return selected_context, selection_scores

    def init_state(self, rnn_state, rnn_output, context, checklist):
        attn, scores = self.compute_attention(rnn_output, context, checklist)
        selected_context, selection_scores = self.select(rnn_output, context[0])
        return (rnn_state, attn, context, selected_context, selection_scores)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            prev_rnn_state, prev_attn, prev_context, selected_context, selection_scores = state
            inputs, checklist = inputs
            # RNN step
            new_inputs = tf.concat(1, [inputs, prev_attn, selected_context])
            output, rnn_state = self.rnn_cell(new_inputs, prev_rnn_state)
            # No update in context inside an utterance
            attn, attn_scores = self.compute_attention(output, prev_context, checklist)
            # Output
            new_output = self.output_with_attention(output, attn)
            return (new_output, attn_scores), (rnn_state, attn, prev_context, selected_context, selection_scores)

