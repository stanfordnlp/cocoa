'''
NN models that take a sequence of words and actions.
Encode when action is read and decode when action is write.
'''

import tensorflow as tf
import numpy as np
from itertools import izip
from tensorflow.python.util import nest
from src.model.rnn_cell import build_rnn_cell
from src.model.util import transpose_first_two_dims, batch_linear, batch_embedding_lookup, EPS

def add_basic_model_arguments(parser):
    parser.add_argument('--model', default='encdec', help='Model name {encdec}')
    parser.add_argument('--rnn-size', type=int, default=20, help='Dimension of hidden units of RNN')
    parser.add_argument('--rnn-type', default='lstm', help='Type of RNN unit {rnn, gru, lstm}')
    parser.add_argument('--num-layers', type=int, default=1, help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of examples per batch')
    parser.add_argument('--word-embed-size', type=int, default=20, help='Word embedding size')
    parser.add_argument('--re-encode', default=False, action='store_true', help='Re-encode the decoded sequence')

def optional_add(feed_dict, key, value):
    if value is not None:
        feed_dict[key] = value

class Sampler(object):
    '''
    Return a symbol from output/logits (batch_size, seq_len, vocab_size).
    '''
    def __init__(self, t, select=None):
        self.t = t  # Temperature
        self.repeat_penalty = 2.
        # If select is not None, we will down weight <select> during sampling
        self.select = select

    def sample(self, logits, prev_words=None, masked_words=None):
        assert logits.shape[1] == 1
        if prev_words is not None:
            prev_words = np.expand_dims(prev_words, 1)
            logits = np.where(prev_words == 1, logits - np.log(2), logits)

        if masked_words is not None:
            for i, words in enumerate(masked_words):
                for word_id in words:
                    logits[i][0][word_id] = float('-inf')

        if self.select is not None:
            logits[:, 0, self.select] -= np.log(2)

        # Greedy
        if self.t == 0:
            return np.argmax(logits, axis=2)
        # Multinomial sample
        else:
            p = self.softmax(logits, self.t)
            batch_size, seq_len, num_symbols = logits.shape
            preds = np.zeros([batch_size, seq_len], dtype=np.int32)
            for i in xrange(batch_size):
                for j in xrange(seq_len):
                    try:
                        preds[i][j] = np.random.choice(num_symbols, 1, p=p[i][j])[0]
                    # p[i][j] do not sum to 1
                    except ValueError:
                        preds[i][j] = np.argmax(p[i][j])
            return preds

    def softmax(self, logits, t=1):
        exp_x = np.exp(logits / t)
        return exp_x / np.sum(exp_x, axis=2, keepdims=True)

class BasicEncoder(object):
    '''
    A basic RNN encoder.
    '''
    def __init__(self, rnn_size, rnn_type='lstm', num_layers=1, dropout=0):
        self.rnn_size = rnn_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.output_dict = {}

    def _build_init_output(self, cell):
        '''
        Initializer for scan. Should have the same shape as the RNN output.
        '''
        return tf.zeros([self.batch_size, cell.output_size])

    def _get_final_state(self, states, last_inds=None):
        '''
        Return the final non-pad state from tf.scan outputs.
        '''
        if last_inds is None:
            last_inds = self.last_inds
        with tf.name_scope(type(self).__name__+'/get_final_state'):
            flat_states = nest.flatten(states)
            flat_last_states = []
            for state in flat_states:
                state = transpose_first_two_dims(state)  # (batch_size, time_seq, state_size)
                # NOTE: when state has dim=4, it's the context which does not change in a seq; just take the last one.
                if len(state.get_shape()) == 4:
                    last_state = state[:, -1, :, :]
                else:
                    last_state = tf.squeeze(batch_embedding_lookup(state, tf.reshape(last_inds, [-1, 1])), [1])
                flat_last_states.append(last_state)
            last_states = nest.pack_sequence_as(states, flat_last_states)
        return last_states

    def _build_rnn_cell(self):
        return build_rnn_cell(self.rnn_type, self.rnn_size, self.num_layers, self.keep_prob)

    def _build_init_state(self, cell, input_dict):
        initial_state = input_dict.get('init_state', None)
        batch_size = input_dict.get('batch_size', self.batch_size)
        if initial_state is not None:
            return initial_state
        else:
            return cell.zero_state(batch_size, tf.float32)

    def get_rnn_inputs_args(self):
        '''
        Return inputs used to build_rnn_inputs for encoding.
        '''
        return {
                'inputs': self.inputs,
                'last_inds': self.last_inds,
                'batch_size': self.batch_size,
                }

    def _build_rnn_inputs(self, time_major, **kwargs):
        word_embedder = self.word_embedder
        inputs = kwargs.get('inputs', self.inputs)

        inputs = word_embedder.embed(inputs, zero_pad=True)
        if not time_major:
            inputs = transpose_first_two_dims(inputs)  # (seq_len, batch_size, input_size)
        return inputs

    def _build_inputs(self, input_dict):
        with tf.name_scope('Inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=[None, None], name='inputs')
            self.last_inds = tf.placeholder(tf.int32, shape=[None], name='last_inds')

    def build_model(self, word_embedder, input_dict, tf_variables, time_major=True, scope=None):
        '''
        inputs: (batch_size, seq_len, input_size)
        '''
        with tf.variable_scope(type(self).__name__):
            self.word_embedder = word_embedder

            self._build_inputs(input_dict)
            self.batch_size = tf.shape(self.inputs)[0]
            self.seq_len = tf.shape(self.inputs)[1]

            cell = self._build_rnn_cell()
            self.cell = cell
            self.init_state = self._build_init_state(cell, input_dict)
            self.output_size = cell.output_size

            inputs = self._build_rnn_inputs(time_major)
            with tf.variable_scope('encode'):
                rnn_outputs, states = tf.scan(lambda a, x: cell(x, a[1]), inputs, initializer=(self._build_init_output(cell), self.init_state))
            self._build_output_dict(rnn_outputs, states)

    def _build_output_dict(self, rnn_outputs, rnn_states):
        final_state = self._get_final_state(rnn_states)
        self.output_dict.update({'outputs': rnn_outputs, 'final_state': final_state})

    def encode(self, time_major=False, **kwargs):
        '''
        Used for additional encoding.
        '''
        inputs = self._build_rnn_inputs(time_major, **kwargs)
        init_state = self._build_init_state(self.cell, kwargs)
        rnn_outputs, states = tf.scan(lambda a, x: self.cell(x, a[1]), inputs, initializer=(self._build_init_output(self.cell), init_state))
        final_state = self._get_final_state(states, kwargs.get('last_inds', self.last_inds))
        final_output = self._get_final_state(rnn_outputs, kwargs.get('last_inds', self.last_inds))
        return {'outputs': rnn_outputs, 'states': states, 'final_state': final_state, 'final_output': final_output}

    def get_feed_dict(self, **kwargs):
        feed_dict = kwargs.pop('feed_dict', {})
        feed_dict[self.inputs] = kwargs.pop('inputs')
        feed_dict[self.last_inds] = kwargs.pop('last_inds')
        feed_dict[self.keep_prob] = 1. - self.dropout
        optional_add(feed_dict, self.init_state, kwargs.pop('init_state', None))
        return feed_dict

    def run(self, sess, fetches, feed_dict):
        results = sess.run([self.output_dict[x] for x in fetches], feed_dict=feed_dict)
        return {k: results[i] for i, k in enumerate(fetches)}

    def run_encode(self, sess, **kwargs):
        feed_dict = self.get_feed_dict(**kwargs)
        return self.run(sess, ('final_state',), feed_dict)

class BasicDecoder(BasicEncoder):
    def __init__(self, rnn_size, num_symbols, rnn_type='lstm', num_layers=1, dropout=0, sample_t=0, sample_select=None, reward=None):
        super(BasicDecoder, self).__init__(rnn_size, rnn_type, num_layers, dropout)
        self.num_symbols = num_symbols
        self.sampler = Sampler(sample_t, sample_select)
        if reward is not None:
            self.add_reward = True
            self.select_penalty = -1. * reward[0]
            self.success_reward = 1. * reward[1]
        else:
            self.add_reward = False

    def get_encoder_state(self, state):
        '''
        Given the hidden state to the encoder to continue from there.
        '''
        return state

    def get_feed_dict(self, **kwargs):
        feed_dict = super(BasicDecoder, self).get_feed_dict(**kwargs)
        optional_add(feed_dict, self.matched_items, kwargs.pop('matched_items', None))
        return feed_dict

    def _build_inputs(self, input_dict):
        super(BasicDecoder, self)._build_inputs(input_dict)
        with tf.name_scope('Inputs'):
            self.matched_items = tf.placeholder(tf.int32, shape=[None], name='matched_items')

    def _build_output(self, output_dict):
        '''
        Take RNN outputs and produce logits over the vocab.
        '''
        outputs = output_dict['outputs']
        outputs = transpose_first_two_dims(outputs)  # (batch_size, seq_len, output_size)
        logits = batch_linear(outputs, self.num_symbols, True)
        #logits = self.penalize_repetition(logits)
        return logits

    @classmethod
    def penalize_repetition(cls, logits):
        #return logits
        exp_logits = tf.exp(tf.clip_by_value(logits, -100, 100))
        logits = tf.log(tf.clip_by_value(exp_logits, 1e-10, 1e10)) - tf.log(tf.clip_by_value((tf.cumsum(exp_logits, axis=1) - exp_logits), 1e-10, 1e10))
        return logits

    # TODO: add a Loss class?
    def compute_loss(self, targets, pad, select):
        logits = self.output_dict['logits']
        loss, seq_loss, total_loss = self._compute_loss(logits, targets, pad)
        if self.add_reward:
            loss += self._compute_penalty(logits, targets, self.matched_items, pad, select, self.select_penalty, self.success_reward)
        # -1 is selection loss
        return loss, seq_loss, total_loss, tf.constant(-1)

    @classmethod
    def _compute_penalty(cls, logits, targets, matched_items, pad, select, select_penalty, success_reward):
        '''
        matched_items: (batch_size,) in the range of num_symbols
        '''
        batch_size = tf.shape(logits)[0]
        num_symbols = tf.shape(logits)[2]
        logprobs = tf.log(tf.nn.softmax(logits) + EPS)
        pad_mask = tf.not_equal(targets, pad)

        correct_items = tf.one_hot(matched_items, num_symbols, on_value=1, off_value=0)  # (batch_size, num_symbols)
        # Pick correct select utterances
        select_utterances = tf.equal(targets[:, 0], select)  # (batch_size,)
        mask = tf.cast(tf.where(select_utterances, correct_items, tf.zeros_like(correct_items)), tf.bool)  # (batch_size, num_symbols)
        item_logprobs = logprobs[:, 1, :]
        correct_item_logprobs = tf.reduce_sum(tf.where(mask, item_logprobs, tf.zeros_like(item_logprobs)), 1)
        success_loss = -1 * success_reward *  correct_item_logprobs

        # Only penalize incorrect select
        select_loss = logprobs[:, 0, select] * select_penalty  # (batch_size,)
        mask = tf.logical_and(select_utterances, tf.equal(targets[:, 1], matched_items))
        select_loss = tf.where(mask, tf.zeros_like(select_loss), select_loss)

        success_loss = tf.where(mask, success_loss, tf.zeros_like(success_loss))

        loss = select_loss + success_loss
        loss = tf.where(pad_mask[:, 0], loss, tf.zeros_like(loss))

        return tf.reduce_sum(loss) / tf.to_float(batch_size)

    @classmethod
    def _compute_loss(cls, logits, targets, pad):
        '''
        logits: (batch_size, seq_len, vocab_size)
        targets: (batch_size, seq_len)
        '''
        batch_size = tf.shape(logits)[0]
        num_symbols = tf.shape(logits)[2]
        # sparse_softmax_cross_entropy_with_logits only takes 2D tensors
        logits = tf.reshape(logits, [-1, num_symbols])
        targets = tf.reshape(targets, [-1])

        # Mask padded tokens
        token_weights = tf.cast(tf.not_equal(targets, tf.constant(pad)), tf.float32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets) * token_weights
        total_loss = tf.reduce_sum(loss)
        token_weights_sum = tf.reduce_sum(tf.reshape(token_weights, [batch_size, -1]), 1) + EPS
        # Average over words in each sequence
        seq_loss = tf.reduce_sum(tf.reshape(loss, [batch_size, -1]), 1) / token_weights_sum

        # Average over sequences
        loss = tf.reduce_sum(seq_loss) / tf.to_float(batch_size)
        # total_loss is used to compute perplexity
        return loss, seq_loss, (total_loss, tf.reduce_sum(token_weights))

    def build_model(self, word_embedder, input_dict, tf_variables, time_major=True, scope=None):
        super(BasicDecoder, self).build_model(word_embedder, input_dict, tf_variables, time_major=time_major, scope=scope)  # outputs: (seq_len, batch_size, output_size)
        with tf.variable_scope(scope or type(self).__name__):
            logits = self._build_output(self.output_dict)
        self.output_dict['logits'] = logits

    def pred_to_input(self, preds, **kwargs):
        '''
        Convert predictions to input of the next decoding step.
        '''
        textint_map = kwargs.pop('textint_map')
        inputs = textint_map.pred_to_input(preds)
        return inputs

    def run_decode(self, sess, max_len, batch_size=1, stop_symbol=None, **kwargs):
        if stop_symbol is not None:
            assert batch_size == 1, 'Early stop only works for single instance'
        feed_dict = self.get_feed_dict(**kwargs)
        preds = np.zeros([batch_size, max_len], dtype=np.int32)
        # last_inds=0 because input length is one from here on
        last_inds = np.zeros([batch_size], dtype=np.int32)
        for i in xrange(max_len):
            logits, final_state = sess.run((self.output_dict['logits'], self.output_dict['final_state']), feed_dict=feed_dict)
            step_preds = self.sampler.sample(logits)
            preds[:, [i]] = step_preds
            if step_preds[0][0] == stop_symbol:
                break
            feed_dict = self.get_feed_dict(inputs=self.pred_to_input(step_preds, **kwargs),
                    last_inds=last_inds,
                    init_state=final_state)
        return {'preds': preds, 'final_state': final_state}

############# dynamic import depending on task ##################
import src.config as config
import importlib
task_module = importlib.import_module('.'.join(('src.model', config.task)))
add_model_arguments = task_module.add_model_arguments
build_model = task_module.build_model
