import tensorflow as tf
from src.model.util import batch_linear
from src.model.encdec import optional_add

# TODO: should probably have a base TFModle class for all TF models, e.g. implements get_feed_dict etc.
class PricePredictor(object):
    '''
    Basic feed-forward NN for price prediction.
    '''
    def __init__(self, hidden_size, history_len):
        self.hidden_size = hidden_size
        self.history_len = history_len

    def _build_inputs(self):
        with tf.name_scope('Inputs'):
            self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.history_len], name='price_history')  # (batch_size, seq_len, input_size)
            self.targets = tf.placeholder(tf.float32, shape=[None, None], name='targets')  # (batch_size, seq_len)

    def build_model(self, context):
        '''
        context: (batch_size, seq_len, context_size)
        '''
        with tf.variable_scope(type(self).__name__):
            inputs = tf.concat(2, [self.inputs, context])  # (batch_size, seq_len, history_len+context_size)
            input_size = self.history_len + context.get_shape().as_list()[-1]
            h = tf.relu(tf.batch_linear(inputs, self.hidden_size, True))  # (batch_size, seq_len, hidden_size)
            with tf.variable_scope('Output'):
                prices = tf.batch_linear(h, 1, True)  # (batch_size, seq_len, 1)
            self.output_dict['prices'] = prices

    def compute_loss(self, pad):
        '''
        MSE loss.
        '''
        targets = self.targets
        preds = self.output_dict['prices']
        weights = tf.cast(tf.not_equal(self.targets, tf.constant(pad)), tf.float32)
        loss = tf.losses.mean_squared_error(targets, preds, weights=weights)
        return loss

    def get_feed_dict(self, **kwargs):
        feed_dict = kwargs.pop('feed_dict', {})
        feed_dict[self.inputs] = kwargs.pop('inputs')
        optional_add(feed_dict, self.targets, kwargs.pop('targets', None))
