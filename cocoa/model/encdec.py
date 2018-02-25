'''
NN models that take a sequence of words and actions.
Encode when action is read and decode when action is write.
'''

import tensorflow as tf
import numpy as np
from itertools import izip
from tensorflow.python.util import nest
from cocoa.model.rnn_cell import build_rnn_cell
from cocoa.model.util import transpose_first_two_dims, batch_linear, batch_embedding_lookup, EPS

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
    parser.add_argument('--decoding', nargs='+', default=['sample', 0, 'select'], help='Decoding method')
    parser.add_argument('--tied', action='store_true', help='Tie weights of word embedding and output')
    parser.add_argument('--stateful', action='store_true', help='Pass final state to next batch')

def optional_add(feed_dict, key, value):
    if value is not None:
        feed_dict[key] = value

class Sampler(object):
    '''
    Return a symbol from output/logits (batch_size, seq_len, vocab_size).
    '''
    def __init__(self, t, select=None, trie=None):
        self.t = t  # Temperature
        self.repeat_penalty = 2.
        # If select is not None, we will down weight <select> during sampling
        self.select = select
        self.trie = trie

    def sample(self, logits, prev_words=None, masked_words=None, prefix=None):
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

        if self.trie is not None and prefix is not None:
            prefix = prefix[:, -3:]  # (batch_size, seq_len)
            mask = np.zeros_like(logits)
            for i, p in enumerate(prefix):
                try:
                    allowed = self.trie.get_children(p)
                    mask[i, :, allowed] = 1
                except KeyError:
                    mask[i, :, :] = 1

            logits = np.where(mask == 1, logits, float('-inf'))

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
                        preds[i][j] = np.where(np.random.multinomial(1, p[i][j]) == 1)[0][0]
                    # p[i][j] do not sum to 1
                    except ValueError:
                        preds[i][j] = np.argmax(p[i][j])
            return preds

    def softmax(self, logits, t=1):
        #exp_x = np.exp(logits / t)
        exp_x = np.exp((logits - np.max(logits)) / t)
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

    def get_inference_args(self, batch, encoder_init_state):
        encoder_args = {'inputs': batch['encoder_args']['inputs'],
                'init_cell_state': encoder_init_state,
                }
        return encoder_args

    def _build_rnn_inputs(self, inputs, input_dict):
        inputs, mask = self.seq_embedder.build_seq_inputs(inputs, self.word_embedder, self.pad, time_major=False)
        init_cell_state = input_dict.get('init_cell_state', None)
        return inputs, mask, {'init_cell_state': init_cell_state}

    def _build_inputs(self, input_dict):
        if 'inputs' in input_dict:
            inputs = input_dict['inputs']
        else:
            inputs = tf.placeholder(tf.int32, shape=[None, None], name='inputs')  # (batch_size, seq_len)
            self.feedable_vars['inputs'] = inputs
        return inputs
        #self.batch_size = tf.shape(self.inputs)[0]
        #self.seq_len = tf.shape(self.inputs)[1]

    def build_model(self, input_dict={}, tf_variables=None):
        '''
        inputs: (batch_size, seq_len, input_size)
        '''
        with tf.variable_scope(type(self).__name__):
            inputs = self._build_inputs(input_dict)

            inputs, mask, kwargs = self._build_rnn_inputs(inputs, input_dict)
            with tf.variable_scope('Embed'):
                embeddings = self.seq_embedder.embed(inputs, mask, **kwargs)

            self._build_output_dict(embeddings)

    def _build_output_dict(self, embeddings):
        self.output_dict.update({
            'outputs': embeddings['step_embeddings'],
            'final_output': embeddings['embedding'],
            'final_state': embeddings['final_state'],
            })

    def get_feed_dict(self, **kwargs):
        feed_dict = kwargs.pop('feed_dict', {})
        if 'inputs' in self.feedable_vars:
            feed_dict[self.feedable_vars['inputs']] = kwargs.pop('inputs')
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
    def __init__(self, word_embedder, seq_embedder, pad, keep_prob, num_symbols, sampler=Sampler(0), sampled_loss=False, tied=False, prompt_len=1):
        super(BasicDecoder, self).__init__(word_embedder, seq_embedder, pad, keep_prob)
        self.num_symbols = num_symbols
        self.sampler = sampler
        self.sampled_loss = sampled_loss
        self.tied = tied
        # Number of symbols before the actual utterance, e.g. <go>
        self.prompt_len = prompt_len

    def get_encoder_state(self, state):
        '''
        Given the hidden state to the encoder to continue from there.
        '''
        return state

    def get_feed_dict(self, **kwargs):
        feed_dict = super(BasicDecoder, self).get_feed_dict(**kwargs)
        optional_add(feed_dict, self.targets, kwargs.pop('targets', None))
        return feed_dict

    def get_inference_args(self, batch, encoder_output_dict, textint_map, prefix_len=1):
        decoder_args = {'inputs': batch['decoder_args']['inputs'][:, :prefix_len],
                'init_cell_state': encoder_output_dict['final_state'],
                'textint_map': textint_map,
                }
        return decoder_args

    def _build_inputs(self, input_dict):
        inputs = super(BasicDecoder, self)._build_inputs(input_dict)
        self.targets = tf.placeholder(tf.int32, shape=[None, None], name='targets')
        return inputs

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
                if self.tied:
                    self.output_w = tf.transpose(self.word_embedder.embedding)
                    self.output_bias = 0
                    inputs = tf.reshape(inputs, [-1, input_size])
                    logits = tf.matmul(inputs, self.output_w) + self.output_bias
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
        tf.summary.scalar('lm_loss', loss)
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

        textint_map = kwargs['textint_map']
        feed_dict = self.get_feed_dict(**kwargs)
        preds = np.zeros([batch_size, max_len], dtype=np.int32)
        inputs = kwargs['inputs']
        for i in xrange(max_len):
            #print '==========%d==========' % i
            logits, final_state = sess.run((self.output_dict['logits'], self.output_dict['final_state']), feed_dict=feed_dict)
            # NOTE: logits might have length > 1 if inputs has > 1 words; take the last one. (batch_size, seq_len, vocab_size)
            logits = logits[:, -1:, :]
            step_preds = self.sampler.sample(logits)
            preds[:, [i]] = step_preds
            if step_preds[0][0] == stop_symbol:
                break
            # TODO: directly change feed_dict item (inputs and init_state)
            feed_dict = self.get_feed_dict(
                    inputs=self.pred_to_input(step_preds, textint_map),
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
