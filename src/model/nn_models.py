'''
Neural network models in keras.
'''

from keras.layers.embeddings import Embedding
from keras.layers import recurrent
from keras.layers.core import Dense
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential

def add_model_arguments(parser):
    parser.add_argument('--model', default='rnnlm', help='Model name {rnnlm}')
    parser.add_argument('--rnn-hidden-size', type=int, default=50, help='Dimension of hidden units of RNN')
    parser.add_argument('--vocab-embed-size', type=int, default=50, help='Dimension of word embedding')
    parser.add_argument('--rnn-type', default='lstm', help='Type of RNN unit {rnn, gru, lstm}')


rnn_layer = {'rnn': recurrent.SimpleRNN,
       'gru': recurrent.GRU,
       'lstm': recurrent.LSTM
      }

class RNNLM(object):
    def __init__(self, vocab_size, embed_size, hidden_size, rnn_type='lstm'):
        RNN = rnn_layer[rnn_type]
        model = Sequential()
        model.add(Embedding(vocab_size, embed_size))
        model.add(RNN(hidden_size, return_sequences=True))
        model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
        self.model = model
