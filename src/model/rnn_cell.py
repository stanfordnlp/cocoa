'''
RNN cell with attention over an input context.
'''

import tensorflow as tf
from src.model.util import linear, batch_linear, batch_embedding_lookup, EPS

rnn_cell = tf.contrib.rnn
recurrent_cell = {'rnn': rnn_cell.BasicRNNCell,
                  'gru': rnn_cell.GRUCell,
                  'lstm': rnn_cell.LSTMCell,
                 }

activation = tf.tanh

def build_rnn_cell(rnn_type, rnn_size, num_layers, keep_prob):
    '''
    Create the internal multi-layer recurrent cell.
    '''
    if rnn_type == 'lstm':
        cell = recurrent_cell[rnn_type](rnn_size, state_is_tuple=True)
    else:
        cell = recurrent_cell[rnn_type](rnn_size)
    if num_layers > 1:
        cell = rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
        cell = rnn_cell.MultiRNNCell([cell] * num_layers)
        cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
    else:
        cell = rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    return cell
