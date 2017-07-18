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
    parser.add_argument('--model', help='Model name {encdec}')
    # TODO: more types of encoder and decoder
    parser.add_argument('--encoder', default='rnn', choices=['rnn'], help='Encoder sequence embedder {bow, rnn}')
    parser.add_argument('--decoder', default='rnn', choices=['rnn', 'rnn-attn'], help='Decoder sequence embedder {rnn, attn}')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate')
    parser.add_argument('--sampled-loss', action='store_true', help='Whether to sample negative examples')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of examples per batch')
    parser.add_argument('--word-embed-size', type=int, default=20, help='Word embedding size')
    parser.add_argument('--pretrained-wordvec', default=None, help='Path to pretrained word embeddings')
    parser.add_argument('--re-encode', default=False, action='store_true', help='Re-encode the decoded sequence')
    parser.add_argument('--decoding', nargs='+', default=['sample', 0, 'select'], help='Decoding method')

def optional_add(feed_dict, key, value):
    if value is not None:
        feed_dict[key] = value

# TODO: fix sampler
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
    A basic RNN encoder for a sequence of inputs.
    '''
    def __init__(self, word_embedder, seq_embedder, pad, keep_prob):
        self.word_embedder = word_embedder
        self.seq_embedder = seq_embedder
        self.pad = pad
        self.keep_prob = keep_prob  # tf.placeholder
        self.output_dict = {}
        self.feedable_vars = {}

    #def _build_init_output(self, cell):
    #    '''
    #    Initializer for scan. Should have the same shape as the RNN output.
    #    '''
    #    return tf.zeros([self.batch_size, cell.output_size])

    #def _get_final_state(self, states, last_inds=None):
    #    '''
    #    Return the final non-pad state from tf.scan outputs.
    #    '''
    #    if last_inds is None:
    #        last_inds = self.last_inds
    #    with tf.name_scope(type(self).__name__+'/get_final_state'):
    #        flat_states = nest.flatten(states)
    #        flat_last_states = []
    #        for state in flat_states:
    #            state = transpose_first_two_dims(state)  # (batch_size, time_seq, state_size)
    #            # NOTE: when state has dim=4, it's the context which does not change in a seq; just take the last one.
    #            if len(state.get_shape()) == 4:
    #                last_state = state[:, -1, :, :]
    #            else:
    #                last_state = tf.squeeze(batch_embedding_lookup(state, tf.reshape(last_inds, [-1, 1])), [1])
    #            flat_last_states.append(last_state)
    #        last_states = nest.pack_sequence_as(states, flat_last_states)
    #    return last_states

    #def _build_rnn_cell(self):
    #    return build_rnn_cell(self.rnn_type, self.rnn_size, self.num_layers, self.keep_prob)

    #def _build_init_state(self, cell, input_dict):
    #    initial_state = input_dict.get('init_state', None)
    #    batch_size = input_dict.get('batch_size', self.batch_size)
    #    if initial_state is not None:
    #        return initial_state
    #    else:
    #        return cell.zero_state(batch_size, tf.float32)

    #def get_rnn_inputs_args(self):
    #    '''
    #    Return inputs used to build_rnn_inputs for encoding.
    #    '''
    #    return {
    #            'inputs': self.inputs,
    #            'last_inds': self.last_inds,
    #            'batch_size': self.batch_size,
    #            }

    def _build_rnn_inputs(self, input_dict):
        inputs = input_dict.get('inputs', self.inputs)
        inputs, mask = self.seq_embedder.build_seq_inputs(inputs, self.word_embedder, self.pad, time_major=False)
        init_cell_state = input_dict.get('init_cell_state', None)
        return inputs, mask, {'init_cell_state': init_cell_state}
        #init_state = input_dict.get('init_state', None)
        #return inputs, mask, {'init_state': init_state}

    def _build_inputs(self, input_dict):
        self.inputs = tf.placeholder(tf.int32, shape=[None, None], name='inputs')  # (batch_size, seq_len)
        #last_inds = tf.reduce_sum(tf.where(tf.equal(self.inputs, pad), tf.zeros_like(self.inputs), tf.ones_like(self.inputs)), 1)
        ## For all-pad inputs
        #last_inds = tf.where(tf.equal(last_inds, 0), tf.ones_like(last_inds), last_inds)
        #self.last_inds = last_inds - 1
        self.batch_size = tf.shape(self.inputs)[0]
        self.seq_len = tf.shape(self.inputs)[1]

    def build_model(self, input_dict={}, tf_variables=None):
        '''
        inputs: (batch_size, seq_len, input_size)
        '''
        with tf.variable_scope(type(self).__name__):
            self._build_inputs(input_dict)

            inputs, mask, kwargs = self._build_rnn_inputs(input_dict)
            with tf.variable_scope('Embed'):
                embeddings = self.seq_embedder.embed(inputs, mask, **kwargs)

            self._build_output_dict(embeddings)

    def _build_output_dict(self, embeddings):
        self.output_dict.update({
            'outputs': embeddings['step_embeddings'],
            'final_state': embeddings['final_state'],
            })

    #def encode(self, time_major=False, **kwargs):
    #    '''
    #    Used for additional encoding.
    #    '''
    #    inputs = self._build_rnn_inputs(time_major, **kwargs)
    #    init_state = self._build_init_state(self.cell, kwargs)
    #    rnn_outputs, states = tf.scan(lambda a, x: self.cell(x, a[1]), inputs, initializer=(self._build_init_output(self.cell), init_state))
    #    final_state = self._get_final_state(states, kwargs.get('last_inds', self.last_inds))
    #    final_output = self._get_final_state(rnn_outputs, kwargs.get('last_inds', self.last_inds))
    #    return {'outputs': rnn_outputs, 'states': states, 'final_state': final_state, 'final_output': final_output}

    def get_feed_dict(self, **kwargs):
        feed_dict = kwargs.pop('feed_dict', {})
        feed_dict[self.inputs] = kwargs.pop('inputs')
        optional_add(feed_dict, self.seq_embedder.feedable_vars['init_cell_state'], kwargs.pop('init_cell_state', None))
        optional_add(feed_dict, self.seq_embedder.feedable_vars['init_state'], kwargs.pop('init_state', None))
        return feed_dict

    def run(self, sess, fetches, feed_dict):
        results = sess.run([self.output_dict[x] for x in fetches], feed_dict=feed_dict)
        return {k: results[i] for i, k in enumerate(fetches)}

    def run_encode(self, sess, **kwargs):
        feed_dict = self.get_feed_dict(**kwargs)
        return self.run(sess, ('final_state', 'outputs'), feed_dict)

class BasicDecoder(BasicEncoder):
    def __init__(self, word_embedder, seq_embedder, pad, keep_prob, num_symbols, sampler=Sampler(0), sampled_loss=False):
        super(BasicDecoder, self).__init__(word_embedder, seq_embedder, pad, keep_prob)
        self.num_symbols = num_symbols
        self.sampler = sampler
        self.sampled_loss = sampled_loss

    def get_encoder_state(self, state):
        '''
        Given the hidden state to the encoder to continue from there.
        '''
        return state

    def get_feed_dict(self, **kwargs):
        feed_dict = super(BasicDecoder, self).get_feed_dict(**kwargs)
        #optional_add(feed_dict, self.feedable_vars['encoder_outputs'], kwargs.pop('encoder_outputs', None))
        optional_add(feed_dict, self.targets, kwargs.pop('targets', None))
        return feed_dict

    def get_inference_args(self, batch, encoder_output_dict, textint_map):
        decoder_args = {'inputs': batch['decoder_inputs'][:, [0]],
                'init_cell_state': encoder_output_dict['final_state'],
                'textint_map': textint_map,
                }
        return decoder_args

    def _build_inputs(self, input_dict):
        super(BasicDecoder, self)._build_inputs(input_dict)
        self.targets = tf.placeholder(tf.int32, shape=[None, None], name='targets')

    def _build_logits(self, inputs):
        '''
        inputs: (batch_size, seq_len, input_size)
        return logits: (batch_size, seq_len)
        '''
        batch_size = tf.shape(inputs)[0]
        input_size = inputs.get_shape().as_list()[-1]
        output_size = self.num_symbols
        # Linear output layer
        with tf.variable_scope('OutputLogits'):
            if self.sampled_loss:
                self.output_w = tf.get_variable('weights', shape=[output_size, input_size], dtype=tf.float32)
                self.output_bias = tf.get_variable('bias', shape=[output_size], dtype=tf.float32)
                inputs = tf.reshape(inputs, [-1, input_size])
                self.inputs_to_output_projection = inputs
                logits = tf.matmul(inputs, tf.transpose(self.output_w)) + self.output_bias
                logits = tf.reshape(logits, [batch_size, -1, output_size])
            else:
                logits = tf.layers.dense(inputs, output_size, activation=None, use_bias=True)
        return logits

    def _build_output(self, output_dict):
        '''
        Take RNN outputs and produce logits over the vocab.
        '''
        inputs = transpose_first_two_dims(output_dict['outputs'])  # (batch_size, seq_len, embed_size)
        logits = self._build_logits(inputs)
        return logits

    def _build_output_dict(self, embeddings):
        super(BasicDecoder, self)._build_output_dict(embeddings)
        self.output_dict['logits'] = self._build_output(self.output_dict)

    def compute_loss(self):
        logits = self.output_dict['logits']
        targets = self.targets
        if not self.sampled_loss:
            loss, seq_loss, total_loss = self._compute_logits_loss(logits, targets, self.pad)
        else:
            loss, seq_loss, total_loss = self._compute_sampled_loss(self.inputs_to_output_projection, targets, self.pad)
        return loss, seq_loss, total_loss

    def _compute_sampled_loss(self, inputs, targets, pad):
        '''
        inputs: (batch_size * seq_len, inputs_size)
        targets: (batch_size, seq_len)
        '''
        batch_size = tf.shape(targets)[0]
        labels = tf.reshape(targets, [-1, 1])
        loss = tf.nn.sampled_softmax_loss(weights=self.output_w, biases=self.output_bias, labels=labels, inputs=self.inputs_to_output_projection, num_sampled=512, num_classes=self.num_symbols, partition_strategy='div')
        return self._mask_loss(loss, tf.reshape(targets, [-1]), pad, batch_size)

    @classmethod
    def _mask_loss(cls, loss, targets, pad, batch_size):
        '''
        loss: 1D (batch_size * seq_len,)
        targets: 1D (batch_size * seq_len)
        '''
        # Mask padded tokens
        token_weights = tf.cast(tf.not_equal(targets, tf.constant(pad)), tf.float32)
        loss = loss * token_weights

        # Loss per seq
        token_weights_sum = tf.reduce_sum(tf.reshape(token_weights, [batch_size, -1]), 1) + EPS
        seq_loss = tf.reduce_sum(tf.reshape(loss, [batch_size, -1]), 1) / token_weights_sum

        # Loss per token
        total_loss = tf.reduce_sum(loss)
        num_tokens = tf.reduce_sum(token_weights)
        loss = total_loss / (num_tokens + EPS)

        # total_loss is used to compute perplexity
        return loss, seq_loss, (total_loss, num_tokens)

    @classmethod
    def _compute_logits_loss(cls, logits, targets, pad):
        '''
        logits: (batch_size, seq_len, vocab_size)
        targets: (batch_size, seq_len)
        '''
        batch_size = tf.shape(targets)[0]
        targets = tf.reshape(targets, [-1])

        # sparse_softmax_cross_entropy_with_logits only takes 2D tensors
        num_symbols = tf.shape(logits)[2]
        logits = tf.reshape(logits, [-1, num_symbols])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)

        return cls._mask_loss(loss, targets, pad, batch_size)

    def pred_to_input(self, preds, textint_map):
        '''
        Convert predictions to input of the next decoding step.
        '''
        inputs = textint_map.pred_to_input(preds)
        return inputs

    def run_decode(self, sess, max_len, batch_size=1, stop_symbol=None, **kwargs):
        if stop_symbol is not None:
            assert batch_size == 1, 'Early stop only works for single instance'

        prefix = kwargs.pop('prefix', None)
        if prefix is not None:
            prefix_args = dict(kwargs)
            prefix_args['inputs'] = prefix
            feed_dict = self.get_feed_dict(**prefix_args)
            final_state = sess.run(self.output_dict['final_state'], feed_dict=feed_dict)
            kwargs['init_state'] = final_state

        feed_dict = self.get_feed_dict(**kwargs)
        preds = np.zeros([batch_size, max_len], dtype=np.int32)
        for i in xrange(max_len):
            #print '==========%d==========' % i
            logits, final_state = sess.run((self.output_dict['logits'], self.output_dict['final_state']), feed_dict=feed_dict)
            step_preds = self.sampler.sample(logits)
            #top_words = np.argsort(p[0][0])[::-1]
            #if i == 0:
            #    for j in xrange(10):
            #        id_ = top_words[j]
            #        print id_, p[0][0][id_], kwargs['textint_map'].vocab.to_word(id_)
            preds[:, [i]] = step_preds
            if step_preds[0][0] == stop_symbol:
                break
            # TODO: directly change feed_dict item (inputs and init_state)
            feed_dict = self.get_feed_dict(
                    inputs=self.pred_to_input(step_preds, **kwargs),
                    init_state=final_state,
                    encoder_outputs=kwargs.get('encoder_outputs', None),
                    context=kwargs.get('context', None))
        return {'preds': preds, 'final_state': final_state}

############# dynamic import depending on task ##################
import src.config as config
import importlib
task_module = importlib.import_module('.'.join(('src.model', config.task)))
add_model_arguments = task_module.add_model_arguments
build_model = task_module.build_model
