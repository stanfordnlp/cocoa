'''
RNN cell with attention over an input context.
'''

import tensorflow as tf
from tensorflow.contrib.seq2seq import AttentionWrapper
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import AttentionMechanism
from src.model.util import linear, batch_linear, batch_embedding_lookup, EPS

tf_rnn = tf.contrib.rnn
recurrent_cell = {'rnn': tf_rnn.BasicRNNCell,
                  'gru': tf_rnn.GRUCell,
                  'lstm': tf_rnn.LSTMCell,
                 }

activation = tf.tanh

# TODO: variational dropout
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
        cell = tf_rnn.DropoutWrapper(cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob, input_size=input_size, variational_recurrent=True, dtype=tf.float32)
    return cell

class MultiAttentionWrapper(AttentionWrapper):
    '''
    Accept multiple attention mechanism.
    '''
    def __init__(self,
                 cell,
                 attention_mechanism,
                 attention_size,
                 cell_input_fn=None,
                 probability_fn=None,
                 output_attention=True,
                 name=None):
        try:
            super(MultiAttentionWrapper, self).__init__(
                                cell,
                                attention_mechanism,
                                attention_size,
                                cell_input_fn=cell_input_fn,
                                probability_fn=probability_fn,
                                output_attention=output_attention,
                                name=name)
        except TypeError:
            for mechanism in attention_mechanism:
                if not isinstance(mechanism, AttentionMechanism):
                    raise TypeError(
                        "attention_mechanism must be a AttentionMechanism, saw type: %s"
                        % type(mechanism).__name__)

    def __call__(self, inputs, state, scope=None):
        if scope is not None:
            raise NotImplementedError("scope not None is not supported")

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state

        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        contexts = []
        for attention_mechanism in self._attention_mechanism:
            score = attention_mechanism(cell_output)
            alignments = self._probability_fn(score)

            # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
            alignments = array_ops.expand_dims(alignments, 1)
            # Context is the inner product of alignments and values along the
            # memory time dimension.
            # alignments shape is
            #   [batch_size, 1, memory_time]
            # attention_mechanism.values shape is
            #   [batch_size, memory_time, attention_mechanism.num_units]
            # the batched matmul is over memory_time, so the output shape is
            #   [batch_size, 1, attention_mechanism.num_units].
            # we then squeeze out the singleton dim.
            context = math_ops.matmul(alignments, self._attention_mechanism.values)
            context = array_ops.squeeze(context, [1])
            contexts.append(context)

        # Concatenate context from different attention machnisums
        context = tf.stack(contexts, axis=2)

        attention = self._attention_layer(
            array_ops.concat([cell_output, context], 1))

        next_state = AttentionWrapperState(
            cell_state=next_cell_state,
            attention=attention)

        if self._output_attention:
          return attention, next_state
        else:
          return cell_output, next_state
