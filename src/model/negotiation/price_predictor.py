import tensorflow as tf
from src.model.util import batch_linear
from src.model.encdec import optional_add

def add_price_predictor_arguments(parser):
    parser.add_argument('--predict-price', action='store_true', default=False, help='Use price predictor')
    parser.add_argument('--price-hist-len', default=3, type=int, help='Length of price history for prediction')
    parser.add_argument('--price-predictor-hidden-size', default=20, type=int)

class PricePredictor(object):
    '''
    Basic feed-forward NN for price prediction.
    '''
    def __init__(self, hidden_size, history_len):
        self.hidden_size = hidden_size
        self.history_len = history_len

    def _build_inputs(self):
        with tf.name_scope('Inputs'):
            self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.history_len], name='inputs')  # (batch_size, seq_len, input_size)
            self.targets = tf.placeholder(tf.float32, shape=[None, None], name='targets')  # (batch_size, seq_len)

    def build_model(self, context):
        '''
        context: (batch_size, seq_len, context_size)
        '''
        with tf.variable_scope(type(self).__name__):
            self._build_inputs()
            # TODO: use tf.layers.Dense
            # NOTE: context comes out from rnn which is
            #inputs = tf.concat([self.inputs, context], 2)  # (batch_size, seq_len, history_len+context_size)
            inputs = self.inputs
            input_size = self.history_len + context.get_shape().as_list()[-1]
            h = tf.nn.relu(batch_linear(inputs, self.hidden_size, True))  # (batch_size, seq_len, hidden_size)
            with tf.variable_scope('Output'):
                prices = batch_linear(h, 1, True)  # (batch_size, seq_len, 1)
                prices = tf.squeeze(prices, 2)

            self.output_dict = {
                    'prices': prices,
                    }

    def compute_loss(self, pad):
        '''
        MSE loss.
        '''
        targets = self.targets
        preds = self.output_dict['prices']
        weights = tf.cast(tf.not_equal(self.targets, tf.constant(float(pad))), tf.float32)
        loss = tf.losses.mean_squared_error(targets, preds, weights=weights)
        return loss

    def get_feed_dict(self, **kwargs):
        feed_dict = kwargs.pop('feed_dict', {})
        feed_dict[self.inputs] = kwargs.pop('inputs')
        optional_add(feed_dict, self.targets, kwargs.pop('targets', None))
        return feed_dict
