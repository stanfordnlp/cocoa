'''
RNN cell with attention over an input context.
'''

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import AttentionMechanism, AttentionWrapper, AttentionWrapperState
from tensorflow.python.ops import array_ops, check_ops, math_ops
from tensorflow.python.framework import ops
from cocoa.model.util import linear, batch_linear, batch_embedding_lookup, EPS
from itertools import izip

tf_rnn = tf.contrib.rnn
recurrent_cell = {'rnn': tf_rnn.BasicRNNCell,
                  'gru': tf_rnn.GRUCell,
                  'lstm': tf_rnn.LSTMCell,
                 }

activation = tf.tanh

def build_rnn_cell(rnn_type, rnn_size, num_layers, keep_prob, input_size=None):
    '''
    Create the internal multi-layer recurrent cell.
    '''
    if rnn_type == 'lstm':
        cell = recurrent_cell[rnn_type](rnn_size, state_is_tuple=True)
    else:
        cell = recurrent_cell[rnn_type](rnn_size)
    if num_layers > 1:
        cell = tf_rnn.DropoutWrapper(cell, input_keep_prob=keep_prob, dtype=tf.float32, input_size=input_size)
        cell = tf_rnn.MultiRNNCell([cell] * num_layers)
        cell = tf_rnn.DropoutWrapper(cell, output_keep_prob=keep_prob, dtype=tf.float32, input_size=input_size)
    else:
        #cell = tf_rnn.DropoutWrapper(cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob, input_size=input_size, variational_recurrent=True, dtype=tf.float32)
        cell = tf_rnn.DropoutWrapper(cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob, dtype=tf.float32)
    return cell

class MultiAttentionMechanisumWrapper(object):
    def __init__(self, attention_mechanisms):
        self.attention_mechanisms = attention_mechanisms
        self.num_mechanisms = len(attention_mechanisms)
        self.alignments_sizes = tf.stack([m.alignments_size for m in self.attention_mechanisms])  # (num_mecha,)
        # We will concatenate alignments of different mechanisums.
        # To separate these alignments later, we need to remember where each alignment start.
        self.alignments_start_inds = tf.cumsum(self.alignments_sizes, exclusive=True)
        self.alignments_size = tf.reduce_sum(self.alignments_sizes)  # ()
        self.batch_size = self.attention_mechanisms[0].batch_size

    def initial_alignments(self, batch_size, dtype):
        a = [m.initial_alignments(batch_size, dtype) for m in self.attention_mechanisms]  # [(batch_size, alignments_size),]
        a = self.combine_alignments(a)
        return a

    def separate_alignments(self, alignments):
        # alignments: (batch_size, \sum_m alignments_size_m)
        multi_alignments = []
        for i in xrange(self.num_mechanisms):
            start = self.alignments_start_inds[i]
            end = start + self.alignments_sizes[i]
            a = alignments[:, start:end]
            multi_alignments.append(a)
        return multi_alignments

    def combine_alignments(self, alignments):
        # alignments: list of (batch_size, alignments_size)
        # return: (batch_size, \sum_m alignments_size_m)
        return tf.concat(alignments, axis=1)

class MultiAttentionWrapper(AttentionWrapper):
    '''
    Accept multiple attention mechanism.
    '''
    def __init__(self,
                 cell,
                 attention_mechanism_list,
                 multi_attention_size,
                 attention_layer_size=None,
                 alignment_history=False,
                 cell_input_fn=None,
                 output_attention=True,
                 initial_cell_state=None,
                 name=None):
        assert len(attention_mechanism_list) > 0
        for mechanism in attention_mechanism_list:
            if not isinstance(mechanism, AttentionMechanism):
                raise TypeError(
                    "attention_mechanism must be a AttentionMechanism, saw type: %s"
                    % type(mechanism).__name__)
        self._multi_attention_size = multi_attention_size

        # Hack: initialize with a single attention mechanism then rewrite class members
        # NOTE: bactch_size consistency is only checed for attention_mechanism_list[0]
        super(MultiAttentionWrapper, self).__init__(
                            cell,
                            attention_mechanism_list[0],
                            attention_layer_size=attention_layer_size,
                            alignment_history=alignment_history,
                            cell_input_fn=cell_input_fn,
                            output_attention=output_attention,
                            initial_cell_state=initial_cell_state,
                            name=name)

        if attention_layer_size is None:
            self._attention_size = self._multi_attention_size

        self._attention_mechanism = MultiAttentionMechanisumWrapper(attention_mechanism_list)

    def call(self, inputs, state):
        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = (
            cell_output.shape[0].value or array_ops.shape(cell_output)[0])
        error_message = (
            "When applying AttentionWrapper %s: " % self.name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and the query (decoder output).  Are you using "
            "the BeamSearchDecoder?  You may need to tile your memory input via "
            "the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with ops.control_dependencies(
            [check_ops.assert_equal(cell_batch_size,
                                    self._attention_mechanism.batch_size,
                                    message=error_message)]):
          cell_output = array_ops.identity(
              cell_output, name="checked_cell_output")

        multi_context = []
        multi_alignments = []
        prev_alignments = self._attention_mechanism.separate_alignments(state.alignments)  # list of (batch_size, alignments_size)
        for attention_mechanism, prev_a in izip(self._attention_mechanism.attention_mechanisms, prev_alignments):
            alignments = attention_mechanism(
                cell_output, previous_alignments=prev_a)

            # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
            expanded_alignments = array_ops.expand_dims(alignments, 1)
            # Context is the inner product of alignments and values along the
            # memory time dimension.
            # alignments shape is
            #   [batch_size, 1, memory_time]
            # attention_mechanism.values shape is
            #   [batch_size, memory_time, attention_mechanism.num_units]
            # the batched matmul is over memory_time, so the output shape is
            #   [batch_size, 1, attention_mechanism.num_units].
            # we then squeeze out the singleton dim.
            attention_mechanism_values = attention_mechanism.values
            context = math_ops.matmul(expanded_alignments, attention_mechanism_values)
            context = array_ops.squeeze(context, [1])

            multi_context.append(context)
            multi_alignments.append(alignments)

        # Combine multiple context
        context = tf.concat(multi_context, axis=1)
        with tf.variable_scope('CombineContext'):
            context = tf.layers.dense(context, self._multi_attention_size, use_bias=False, activation=tf.nn.relu)

        # Combine alignments
        alignments = self._attention_mechanism.combine_alignments(multi_alignments)  # (batch_size, \sum_{m} alignments_size_m)

        if self._attention_layer is not None:
          attention = self._attention_layer(
              array_ops.concat([cell_output, context], 1))
        else:
          attention = context

        if self._alignment_history:
          alignment_history = state.alignment_history.write(
              state.time, alignments)
        else:
          alignment_history = ()

        next_state = AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            alignments=alignments,
            alignment_history=alignment_history)

        if self._output_attention:
          return attention, next_state
        else:
          return cell_output, next_state
