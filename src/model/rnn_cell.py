'''
RNN cell with attention over an input context.
'''

import tensorflow as tf
from src.model.util import linear, batch_linear, batch_embedding_lookup, EPS

tf_rnn = tf.contrib.rnn
recurrent_cell = {'rnn': tf_rnn.BasicRNNCell,
                  'gru': tf_rnn.GRUCell,
                  'lstm': tf_rnn.LSTMCell,
                 }

activation = tf.tanh

# TODO: variational dropout
def build_rnn_cell(rnn_type, rnn_size, num_layers, keep_prob):
    '''
    Create the internal multi-layer recurrent cell.
    '''
    if rnn_type == 'lstm':
        cell = recurrent_cell[rnn_type](rnn_size, state_is_tuple=True)
    else:
        cell = recurrent_cell[rnn_type](rnn_size)
    if num_layers > 1:
        cell = tf_rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
        cell = tf_rnn.MultiRNNCell([cell] * num_layers)
        cell = tf_rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    else:
        cell = tf_rnn.DropoutWrapper(cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    return cell
