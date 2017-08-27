import numpy as np
import tensorflow as tf
from cocoa.model.util import batch_linear, transpose_first_two_dims
from cocoa.model.encdec import optional_add

def add_price_predictor_arguments(parser):
    parser.add_argument('--predict-price', action='store_true', default=False, help='Use price predictor')
    parser.add_argument('--price-hist-len', default=1, type=int, help='Length of price history for prediction for each agent')
    parser.add_argument('--price-predictor-hidden-size', default=20, type=int)

class PricePredictor(object):
    '''
    Basic feed-forward NN for price prediction.
    '''
    def __init__(self, hidden_size, history_len, pad):
        self.hidden_size = hidden_size
        self.history_len = history_len
        self.pad = pad
        self._build_inputs()

    def _build_inputs(self):
        with tf.variable_scope(type(self).__name__):
            # Price history for agent and partner. 0: agent (self), 1: partner.
            self.init_price = tf.placeholder(tf.float32, shape=[None, 2, self.history_len], name='init_price_vector')

    def zero_init_price(self, batch_size):
        return np.zeros([batch_size, 2, self.history_len])

    def _update_agent_price(self, curr_price, new_price, agent):
        '''
        curr_price: (batch_size, 2, history_len)
        new_price: (batch_size,)
        '''
        agent_curr_price = curr_price[:, agent, :]
        pad_price = tf.equal(new_price, self.pad)
        # Shift left for new price
        price_before_last_step = tf.where(pad_price, agent_curr_price[:, :-1], agent_curr_price[:, 1:])
        # Append new price or keep last step price
        price_last_step = tf.reshape(tf.where(pad_price, agent_curr_price[:, -1], new_price), [-1, 1])
        agent_price = tf.concat([price_before_last_step, price_last_step], axis=1)
        # Combine self and partner prices
        partner_price = curr_price[:, 1 - agent, :]
        if agent == 0:
            price = tf.reshape(tf.concat([agent_price, partner_price], axis=1), [-1, 2, self.history_len])
        else:
            price = tf.reshape(tf.concat([partner_price, agent_price], axis=1), [-1, 2, self.history_len])
        return price

    #def _update_price(self, curr_price, new_price):
    #    '''
    #    curr_price: (batch_size, 2, history_len)
    #    new_price: (batch_size,)
    #    partner: (), boolean scalar.
    #    '''
    #    new_price, partner = new_price
    #    price = tf.cond(partner,
    #            lambda : self._update_agent_price(curr_price, new_price, 1),
    #            lambda : self._update_agent_price(curr_price, new_price, 0))
    #    return price

    def _update_self_price(self, curr_price, new_price):
        return self._update_agent_price(curr_price, new_price, 0)

    def _update_partner_price(self, curr_price, new_price):
        return self._update_agent_price(curr_price, new_price, 1)

    def update_price(self, partner, inputs, init_price=None):
        '''
        Update price history given the inputs:
            inputs = [[1, 1]]
            partner_input = False
            init_price[:, 0, :] = [[0, 2], [1, 3]]
        Return updated price vector:
            [[2, 1], [3, 1]]
        '''
        # Change to time major
        inputs = transpose_first_two_dims(inputs)
        update_func = self._update_partner_price if partner else self._update_self_price
        if init_price is None:
            init_price = self.init_price
        price_hists = tf.scan(update_func, inputs, initializer=init_price)
        return price_hists

    def predict_price(self, inputs, context):
        '''
        context: (batch_size, seq_len, context_size)
        '''
        with tf.variable_scope(type(self).__name__):
            # Concat feature of self and agent
            inputs = tf.reshape(inputs, [-1, 2 * self.history_len])
            # Repeat along the time dimension
            seq_len = tf.shape(context)[1]
            inputs = tf.tile(tf.expand_dims(inputs, 1), [1, seq_len, 1])  # (batch_size, seq_len, 2*hist_len)

            # NOTE: context comes out from the decoder rnn
            inputs = tf.concat([inputs, context], 2)  # (batch_size, seq_len, history_len+context_size)

            # MLP
            h = tf.layers.dense(inputs, self.hidden_size, tf.nn.tanh)  # (batch_size, seq_len, hidden_size)
            prices = tf.squeeze(tf.layers.dense(h, 1, None), 2)  # (batch_size, seq_len)

        return prices

    def compute_loss(self, preds, targets):
        '''
        MSE loss.
        '''
        weights = tf.cast(tf.not_equal(targets, tf.constant(float(self.pad))), tf.float32)
        #targets = tf.Print(targets, [targets * weights], message='targets: ', summarize=100)
        #preds = tf.Print(preds, [preds * weights], message='preds: ', summarize=100)
        loss = tf.losses.mean_squared_error(targets, preds, weights=weights)
        return loss

    def get_feed_dict(self, **kwargs):
        feed_dict = kwargs.pop('feed_dict', {})
        optional_add(feed_dict, self.init_price, kwargs.pop('init_price_history', None))
        return feed_dict
