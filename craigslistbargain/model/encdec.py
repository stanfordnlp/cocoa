from itertools import izip
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple

from cocoa.model.util import EPS, tile_tensor
from cocoa.model.util import transpose_first_two_dims, batch_embedding_lookup, EPS
from cocoa.model.encdec import BasicEncoder, BasicDecoder, Sampler, optional_add
from cocoa.model.sequence_embedder import AttentionRNNEmbedder, BoWEmbedder

from preprocess import markers, START_PRICE
from price_buffer import PriceBuffer

def add_model_arguments(parser):
    parser.add_argument('--attention-memory', nargs='*', default=None, help='Attention memory: title, description, encoder outputs')
    parser.add_argument('--num-context', default=0, type=int, help='Number of sentences to consider as dialogue context (in addition to the encoder input)')
    parser.add_argument('--selector', action='store_true', help='Retrieval-based model (candidate selector)')
    parser.add_argument('--selector-loss', default='binary', choices=['binary'], help='Loss function for the selector (binary: cross-entropy loss of classifying a candidate as the true response)')

class LM(object):
    def __init__(self, decoder, pad):
        self.decoder = decoder
        self.pad = pad
        self.decoder.build_model()
        self.final_state = self.decoder.output_dict['final_state']
        self.loss, self.seq_loss, self.total_loss = self.decoder.compute_loss()
        self.perplexity = True
        self.name = 'lm'

    def get_feed_dict(self, **kwargs):
        feed_dict = kwargs.pop('feed_dict', {})
        feed_dict = self.decoder.get_feed_dict(feed_dict=feed_dict, **kwargs)
        return feed_dict

    def generate(self, sess, batch, init_state, max_len=100, textint_map=None):
        decoder_args = {'prefix': batch['encoder_args']['inputs'],
                'inputs': batch['decoder_args']['inputs'][:, [0]],
                'init_state': init_state,
                'textint_map': textint_map,
                'context': batch['decoder_args']['context'],
                }
        batch_size = batch['encoder_args']['inputs'].shape[0]
        decoder_output_dict = self.decoder.run_decode(sess, max_len, batch_size, **decoder_args)

        # Go over the true sequence
        # TODO: move prefix to tf graph
        decoder_args = {
                'inputs': batch['encoder_inputs'],
                'init_state': init_state,
                'context': batch['context'],
                }
        feed_dict = self.decoder.get_feed_dict(**decoder_args)
        init_state = sess.run(self.final_state, feed_dict=feed_dict)
        decoder_args = {
                'inputs': batch['decoder_inputs'],
                'init_state': init_state,
                'context': batch['context'],
                }
        feed_dict = self.decoder.get_feed_dict(**decoder_args)
        true_final_state = sess.run(self.final_state, feed_dict=feed_dict)

        return {'preds': decoder_output_dict['preds'],
                'true_final_state': true_final_state,
                }

class BasicEncoderDecoder(object):
    '''
    Basic seq2seq model.
    '''
    def __init__(self, encoder, decoder, pad, keep_prob, stateful=False):
        self.pad = pad  # Id of PAD in the vocab
        self.encoder = encoder
        self.decoder = decoder
        self.tf_variables = set()
        self.perplexity = True
        self.name = 'encdec'
        self.keep_prob = keep_prob
        # stateful: initialize state with state from the previous batch
        self.stateful = stateful
        self.build_model(encoder, decoder)

    def compute_loss(self, output_dict):
        return self.decoder.compute_loss()

    def _encoder_input_dict(self):
        return {
                'init_cell_state': None,
               }

    def _decoder_input_dict(self, encoder_output_dict):
        # TODO: clearner way to add price_history - put this in the state,
        # maybe just return encoder_output_dict...
        return {
                'init_cell_state': encoder_output_dict['final_state'],
                'encoder_embeddings': encoder_output_dict['outputs'],
                'price_history': encoder_output_dict.get('price_history', None),
               }

    def build_model(self, encoder, decoder):
        with tf.variable_scope(type(self).__name__):
            # Encoding
            encoder_input_dict = self._encoder_input_dict()
            encoder.build_model(encoder_input_dict, self.tf_variables)

            # Decoding
            decoder_input_dict = self._decoder_input_dict(encoder.output_dict)
            decoder.build_model(decoder_input_dict, self.tf_variables)

            self.final_state = decoder.get_encoder_state(decoder.output_dict['final_state'])

            # Loss
            # TODO: remove seq_loss
            self.loss, self.seq_loss, self.total_loss = self.compute_loss(decoder.output_dict)

    def get_feed_dict(self, **kwargs):
        feed_dict = kwargs.pop('feed_dict', {})
        if 'encoder' in kwargs:
            feed_dict = self.encoder.get_feed_dict(feed_dict=feed_dict, **kwargs.pop('encoder'))
        if 'decoder' in kwargs:
            feed_dict = self.decoder.get_feed_dict(feed_dict=feed_dict, **kwargs.pop('decoder'))
        return feed_dict

    def output_to_preds(self, logits):
        '''
        logits: (batch_size, seq_len, vocab_size)
        '''
        return np.argmax(logits, axis=2)

    def generate(self, sess, batch, encoder_init_state, max_len=100, textint_map=None, true_inputs=None):
        batch_size = batch['encoder_args']['inputs'].shape[0]

        # Encode true prefix
        encoder_args = self.encoder.get_inference_args(batch, encoder_init_state)
        encoder_output_dict = self.encoder.run_encode(sess, **encoder_args)

        # Decode max_len steps
        decoder_args = self.decoder.get_inference_args(batch, encoder_output_dict, textint_map, prefix_len=self.decoder.prompt_len)
        decoder_output_dict = self.decoder.run_decode(sess, max_len, batch_size, **decoder_args)

        # If model is stateful, we need to pass the last state to the next batch.
        # Therefore we need to decode true utterances to condition on true prefix.
        if self.stateful:
            if true_inputs is None:
                true_inputs = batch['decoder_inputs']['inputs']
            decoder_args['inputs'] = true_inputs
            feed_dict = self.decoder.get_feed_dict(**decoder_args)
            true_final_state = sess.run(self.final_state, feed_dict=feed_dict)
        else:
            true_final_state = None

        return {'preds': decoder_output_dict['preds'],
                'prices': decoder_output_dict.get('prices', None),
                'final_state': decoder_output_dict['final_state'],
                'true_final_state': true_final_state,
                }


class IRSelector(object):
    """Simple candidate selector entirely based on the IR system (tf-idf).
    """
    def __init__(self):
        self.name = 'ir'

    def generate(self, batch):
        candidate_ranks = [[0] + [1 for _ in xrange(len(candidates)-1)] for candidates in batch['token_candidates']]
        responses = [candidates[0] if len(candidates) > 0 else [] for candidates in batch['token_candidates']]
        return candidate_ranks, responses

class CandidateSelector(BasicEncoderDecoder):
    """Learned candidate selector.
    """
    def __init__(self, encoder, decoder, pad, keep_prob):
        '''
        decoder: encode the reponse; doesn't actually decoder word by word.
        '''
        self.pad = pad  # Id of PAD in the vocab
        self.encoder = encoder
        self.decoder = decoder
        self.name = 'selector'
        self.keep_prob = keep_prob
        self.stateful = False
        self.perplexity = False
        self.build_model(encoder, decoder)

    def build_model(self, encoder, decoder):
        with tf.variable_scope(type(self).__name__):
            # Encoding
            encoder_input_dict = self._encoder_input_dict()
            encoder.build_model(encoder_input_dict)

            # Decoding
            decoder_input_dict = self._decoder_input_dict(encoder.output_dict)
            decoder.build_model(decoder_input_dict)

            # Loss
            self.loss, self.total_loss = self.compute_loss(decoder.output_dict)

    def output_to_preds(self, scores):
        '''
        scores: (batch_size, num_candidates)
        '''
        # -1.*: reverse order (descent)
        return np.argsort(-1.*scores, axis=1)

    def select(self, sess, feed_dict_args):
        feed_dict = self.get_feed_dict(**feed_dict_args)
        scores = sess.run(self.decoder.output_dict['scores'], feed_dict=feed_dict)
        # Padded candidates
        candidates = feed_dict_args['decoder']['candidates']
        padded_candidates = candidates[:, :, 0] == self.pad  # (batch_size, num_candidates)
        scores[padded_candidates] = float('-inf')
        candidate_ranks = self.output_to_preds(scores)
        best_candidates = np.argmax(scores, axis=1)
        return best_candidates, candidate_ranks

    def generate(self, sess, batch, init_state, max_len=None, textint_map=None):
        encoder_args = batch['encoder_args']
        encoder_args['init_state'] = init_state
        decoder_args = batch['decoder_args']
        decoder_args['mask'] = batch.get('mask', None)
        kwargs = {'encoder': encoder_args,
                'decoder': decoder_args,
                }
        best_candidates, candidate_ranks = self.select(sess, kwargs)
        batch_candidates = batch['token_candidates']
        responses = [candidates[id_] if id_ < len(candidates) else [] for candidates, id_ in izip(batch_candidates, best_candidates)]
        return {
                'candidate_ranks': candidate_ranks,
                'responses': responses,
                }


class ContextEncoder(BasicEncoder):
    '''
    Encode previous utterances as a context vector which is combined with the current hidden state.
    '''
    def __init__(self, word_embedder, seq_embedder, num_context, pad, keep_prob):
        super(ContextEncoder, self).__init__(word_embedder, seq_embedder, pad, keep_prob)
        self.context_embedder = BoWEmbedder(word_embedder=word_embedder)
        self.num_context = num_context

    def get_inference_args(self, batch, init_state):
        encoder_args = super(ContextEncoder, self).get_inference_args(batch, init_state)
        encoder_args['context'] = batch['encoder_args']['context']
        return encoder_args

    def _build_inputs(self, input_dict):
        inputs = super(ContextEncoder, self)._build_inputs(input_dict)
        # TODO: context
        self.context = [tf.placeholder(tf.int32, shape=[None, None], name='context_%d' % i) for i in xrange(self.num_context)]  # (batch_size, context_seq_len)
        return inputs

    def embed_context(self):
        embeddings = []
        for i in xrange(self.num_context):
            inputs, mask = self.context_embedder.build_seq_inputs(self.context[i], self.word_embedder, self.pad, time_major=False)
            embeddings.append(self.context_embedder.embed(inputs, mask, integer=False)['embedding'])
        embeddings = tf.stack(embeddings)  # (num_context, batch_size, embed_size)
        embeddings = transpose_first_two_dims(embeddings)
        embeddings = tf.reduce_sum(embeddings, axis=1)
        return embeddings  # (batch_size, embed_size)

    def build_model(self, input_dict={}, tf_variables=None):
        '''
        inputs: (batch_size, seq_len, input_size)
        '''
        with tf.variable_scope(type(self).__name__):
            inputs = self._build_inputs(input_dict)

            inputs, mask, kwargs = self._build_rnn_inputs(inputs, input_dict)
            with tf.variable_scope('Embed'):
                embeddings = self.seq_embedder.embed(inputs, mask, **kwargs)
            with tf.variable_scope('ContextEmbed'):
                context_embedding = self.embed_context()

            self._build_output_dict(embeddings, context_embedding)

    def _build_output_dict(self, embeddings, context_embedding):
        super(ContextEncoder, self)._build_output_dict(embeddings)
        final_state = self.output_dict['final_state']
        # Combine context_embedding with final states
        with tf.variable_scope('EncoderState2DecoderState'):
            # TODO: non-LSTM cells
            state_c = tf.layers.dense(
                    tf.concat([final_state.c, context_embedding], axis=1),
                    self.seq_embedder.embed_size,
                    use_bias=False,
                    activation=tf.nn.relu
                    )
            state_h = tf.layers.dense(
                    tf.concat([final_state.h, context_embedding], axis=1),
                    self.seq_embedder.embed_size,
                    use_bias=False,
                    activation=tf.nn.relu
                    )
            state = LSTMStateTuple(state_c, state_h)
            self.output_dict['final_state'] = state

    def get_feed_dict(self, **kwargs):
        feed_dict = super(ContextEncoder, self).get_feed_dict(**kwargs)
        feed_dict.update({i: d for i, d in izip(self.context, kwargs['context'])})
        return feed_dict

    def run_encode(self, sess, **kwargs):
        feed_dict = self.get_feed_dict(**kwargs)
        return self.run(sess, ('final_state', 'outputs'), feed_dict)

class AttentionDecoder(BasicDecoder):
    '''
    Attend to encoder embeddings and/or context.
    '''
    def __init__(self, word_embedder, seq_embedder, pad, keep_prob, num_symbols, sampler=Sampler(0), sampled_loss=False, context_embedder=None, attention_memory=('title',), prompt_len=1):
        assert isinstance(seq_embedder, AttentionRNNEmbedder)
        super(AttentionDecoder, self).__init__(word_embedder, seq_embedder, pad, keep_prob, num_symbols, sampler, sampled_loss, prompt_len=prompt_len)
        self.context_embedder = context_embedder
        context = tuple([x for x in attention_memory if x in self.context_embedder.context_names])
        self.context_embedding = self.context_embedder.embed(context=context, step=True)  # [(batch_size, context_len, embed_size)]

    def get_encoder_state(self, state):
        return state.cell_state

    def get_inference_args(self, batch, encoder_output_dict, textint_map, prefix_len=1):
        decoder_args = super(AttentionDecoder, self).get_inference_args(batch, encoder_output_dict, textint_map, prefix_len=prefix_len)
        decoder_args.update({
            'context': batch['decoder_args']['context'],
            'encoder_outputs': encoder_output_dict['outputs'],
            })
        return decoder_args

    def _build_rnn_inputs(self, inputs, input_dict):
        inputs, mask, kwargs = super(AttentionDecoder, self)._build_rnn_inputs(inputs, input_dict)
        encoder_outputs = input_dict['encoder_embeddings']
        self.feedable_vars['encoder_outputs'] = encoder_outputs
        encoder_embeddings = transpose_first_two_dims(encoder_outputs)  # (batch_size, seq_len, embed_size)
        attention_memory = self.context_embedding + [encoder_embeddings]
        kwargs['attention_memory'] = attention_memory
        # mask doesn't seem to matter
        #kwargs['attention_mask'] = self.context_embedder.get_mask('title')
        return inputs, mask, kwargs

    def get_feed_dict(self, **kwargs):
        feed_dict = super(AttentionDecoder, self).get_feed_dict(**kwargs)
        optional_add(feed_dict, self.feedable_vars['encoder_outputs'], kwargs.pop('encoder_outputs', None))
        feed_dict = self.context_embedder.get_feed_dict(feed_dict=feed_dict, **kwargs.pop('context'))
        return feed_dict

class ContextDecoder(BasicDecoder):
    '''
    Add a context vector (category, title, description) to each decoding step.
    '''
    def __init__(self, word_embedder, seq_embedder, context_embedder, context, pad, keep_prob, num_symbols, sampler=Sampler(0), sampled_loss=False, tied=False):
        super(ContextDecoder, self).__init__(word_embedder, seq_embedder, pad, keep_prob, num_symbols, sampler, sampled_loss, tied)
        self.context_embedder = context_embedder
        #self.context = context
        self.context_embedding = self.context_embedder.embed(context)

    def _build_rnn_inputs(self, inputs, input_dict):
        inputs, mask, kwargs = super(ContextDecoder, self)._build_rnn_inputs(inputs, input_dict)  # (seq_len, batch_size, input_size)
        inputs = self.seq_embedder.concat_vector_to_seq(self.context_embedding, inputs)
        return inputs, mask, kwargs

    def get_inference_args(self, batch, encoder_output_dict, textint_map, prefix_len=1):
        decoder_args = super(ContextDecoder, self).get_inference_args(batch, encoder_output_dict, textint_map, prefix_len=prefix_len)
        decoder_args['context'] = batch['decoder_args']['context']
        return decoder_args

    def get_feed_dict(self, **kwargs):
        feed_dict = super(ContextDecoder, self).get_feed_dict(**kwargs)\
        # TODO: context should always be in feed_dict. however sometimes we want to update
        # values in the feed_dict and context might have already been there. should
        # check that it's true though.
        if 'context' in kwargs:
            feed_dict = self.context_embedder.get_feed_dict(feed_dict=feed_dict, **kwargs.pop('context'))
        return feed_dict

class PriceEncoder(object):
    '''
    A wrapper of a encoder that uses a price predictor to update prices.
    '''
    def __init__(self, encoder, price_predictor):
        self.encoder = encoder
        self.price_predictor = price_predictor

    def build_model(self, input_dict, tf_variables):
        with tf.variable_scope(type(self).__name__):
            self.encoder.build_model(input_dict, tf_variables)
            self.price_inputs = tf.placeholder(tf.float32, shape=[None, None], name='price_inputs')  # (batch_size, seq_len)
            # Update price. partner = True. Take the price at the last time step.
            new_price_history = self.price_predictor.update_price(True, self.price_inputs)[-1]

            # Outputs
            self.output_dict = dict(self.encoder.output_dict)
            self.output_dict['price_history'] = new_price_history

    def get_feed_dict(self, **kwargs):
        feed_dict = self.encoder.get_feed_dict(**kwargs)
        feed_dict[self.price_inputs] = kwargs.pop('price_inputs')
        feed_dict = self.price_predictor.get_feed_dict(feed_dict=feed_dict, **kwargs.pop('price_predictor'))
        return feed_dict

class DecoderWrapper(object):
    '''
    Wrapper around a decoder.
    '''
    def __init__(self, decoder):
        self.decoder = decoder
        self.sampler = decoder.sampler

    def build_model(self, *args, **kwargs):
        with tf.variable_scope(type(self).__name__):
            self.decoder.build_model(*args, **kwargs)
        self.output_dict = dict(self.decoder.output_dict)

    def get_inference_args(self, *args):
        return self.decoder.get_inference_args(*args)

    def get_feed_dict(self, **kwargs):
        return self.decoder.get_feed_dict(**kwargs)

    def pred_to_input(self, preds, textint_map):
        return self.decoder.pred_to_input(preds, textint_map)

    def get_encoder_state(self, state):
        return self.decoder.get_encoder_state(state)

    def compute_loss(self):
        return self.decoder.compute_loss()

class TrieDecoder(DecoderWrapper):
    def build_model(self, *args, **kwargs):
        super(TrieDecoder, self).build_model(*args, **kwargs)
        logits = self.output_dict['logits']
        mask = tf.placeholder(tf.bool, shape=logits.get_shape().as_list(), name='mask')
        self.decoder.feedable_vars['mask'] = mask
        masked_logits = tf.where(mask, logits, -10.*tf.ones_like(logits))
        #masked_logits = logits + tf.where(mask, 1.*tf.ones_like(logits), -5.*tf.ones_like(logits))
        self.decoder.output_dict['logits'] = masked_logits
        self.output_dict['logits'] = masked_logits

    def get_inference_args(self, batch, encoder_output_dict, textint_map, prefix_len=1):
        args = super(TrieDecoder, self).get_inference_args(*args)
        args['mask'] = batch['mask'][:, :prefix_len, :]
        args['mask'] = np.ones_like(args['mask'])
        return args

    def get_feed_dict(self, **kwargs):
        feed_dict = super(TrieDecoder, self).get_feed_dict(**kwargs)
        feed_dict[self.decoder.feedable_vars['mask']] = kwargs['mask']
        return feed_dict

class SlotFillingDecoder(DecoderWrapper):
    def _fill_in_slots(self, sess, feed_dict, curr_state, init_input, preds, textint_map, stop_symbol=None, max_len=10):
        feed_dict = self.get_feed_dict(feed_dict=feed_dict, inputs=init_input, init_state=curr_state)
        preds.append(init_input)
        for i in xrange(max_len):
            logits, curr_state = sess.run((self.output_dict['logits'], self.output_dict['final_state']), feed_dict=feed_dict)
            step_preds = self.sampler.sample(logits)
            preds.append(step_preds)
            if step_preds[0][0] == stop_symbol:
                break
            # Update inputs and init_state
            feed_dict = self.get_feed_dict(
                    inputs=self.pred_to_input(step_preds, textint_map),
                    init_state=curr_state,
                    feed_dict=feed_dict)
        return curr_state

    def _read_prefix(self, sess, feed_dict, curr_state, prefix, preds):
        feed_dict = self.get_feed_dict(feed_dict=feed_dict, inputs=prefix, init_state=curr_state)
        preds.append(prefix)
        curr_state = sess.run(self.output_dict['final_state'], feed_dict=feed_dict)
        return curr_state

    def get_inference_args(self, batch, encoder_output_dict, textint_map, prefix_len=1):
        decoder_args = super(SlotFillingDecoder, self).get_inference_args(batch, encoder_output_dict, textint_map, prefix_len=prefix_len)
        decoder_args['inputs'] = batch['decoder_args']['inputs']
        return decoder_args

    def run_decode(self, sess, max_len=10, batch_size=1, **kwargs):
        assert batch_size == 1
        textint_map = kwargs['textint_map']
        end_slot = textint_map.vocab.to_ind(markers.END_SLOT)
        start_slot = textint_map.vocab.to_ind(markers.START_SLOT)
        feed_dict = self.get_feed_dict(**kwargs)
        inputs = kwargs['inputs']
        N = inputs.shape[1]
        slot_pos = np.where((inputs == start_slot) | (inputs == end_slot))[1].tolist()
        slot_pos.insert(0, -1)
        slot_pos.append(N)
        iter_slot_pos = iter(slot_pos)
        preds = []
        curr_state = None
        def to_str(a):
            return kwargs['textint_map'].int_to_text(list(a))
        for prev_slot_end, curr_slot_start in izip(iter_slot_pos, iter_slot_pos):
            # Go through prefix
            prefix = inputs[:, prev_slot_end+1:curr_slot_start]
            if prefix.shape[1] > 0:
                curr_state = self._read_prefix(sess, feed_dict, curr_state, prefix, preds)

            # Start to generate
            if curr_slot_start < N:
                init_input = inputs[:, [curr_slot_start]]  # START_SLOT
                curr_state = self._fill_in_slots(sess, feed_dict, curr_state, init_input, preds, textint_map, stop_symbol=end_slot, max_len=max_len)

        preds = np.concatenate(tuple(preds), axis=1)
        return {'preds': preds, 'final_state': curr_state}

class PriceDecoder(object):
    '''
    A wrapper of a decoder that outputs <price> and a price predictor that fills in the actual price.
    '''
    def __init__(self, decoder, price_predictor):
        self.decoder = decoder
        self.price_predictor = price_predictor

    def get_inference_args(self, batch, encoder_output_dict, textint_map, prefix_len=1):
        decoder_args = self.decoder.get_inference_args(batch, encoder_output_dict, textint_map, prefix_len=prefix_len)
        price_args = {'inputs': batch['decoder_args']['price_inputs'][:, [0]]}
        decoder_args['price_predictor'] = price_args
        decoder_args['price_symbol'] = textint_map.vocab.to_ind('<price>'),
        return decoder_args

    def build_model(self, input_dict, tf_variables):
        with tf.variable_scope(type(self).__name__):
            self.decoder.build_model(input_dict, tf_variables)
            # NOTE: output from rnn is time major
            # context: hidden states at each time step
            context = transpose_first_two_dims(self.decoder.output_dict['outputs'])
            self.price_inputs = tf.placeholder(tf.float32, shape=[None, None], name='price_inputs')  # (batch_size, seq_len)
            self.price_targets = tf.placeholder(tf.float32, shape=[None, None], name='price_targets')  # (batch_size, seq_len)
            init_price = input_dict['price_history']  # (batch_size, price_size)
            # NOTE: no price updating during decoding
            predicted_prices = self.price_predictor.predict_price(init_price, context)
            # Update price after decoding. partner = False
            new_price_history_seq = self.price_predictor.update_price(False, self.price_targets, init_price=init_price)

            # Outputs
            self.output_dict = dict(self.decoder.output_dict)
            self.output_dict['price_history'] = new_price_history_seq[-1, :, :]
            self.output_dict['price_preds'] = predicted_prices

    def compute_loss(self):
        loss, seq_loss, total_loss = self.decoder.compute_loss()
        price_loss = self.price_predictor.compute_loss(self.output_dict['price_preds'], self.price_targets)
        loss += price_loss
        # NOTE: seq_loss and total_loss do not depend on price_loss. We're using loss for bp.
        tf.summary.scalar('price_loss', price_loss)
        return loss, seq_loss, total_loss

    def get_feed_dict(self, **kwargs):
        feed_dict = self.decoder.get_feed_dict(**kwargs)
        feed_dict[self.price_inputs] = kwargs.pop('price_inputs')
        optional_add(feed_dict, self.price_targets, kwargs.pop('price_targets', None))
        feed_dict = self.price_predictor.get_feed_dict(feed_dict=feed_dict, **kwargs.pop('price_predictor'))
        return feed_dict

    def get_encoder_state(self, state):
        '''
        Given the hidden state to the encoder to continue from there.
        '''
        return self.decoder.get_encoder_state(state)

    def pred_to_input(self, preds, textint_map):
        '''
        Convert predictions to input of the next decoding step.
        '''
        inputs = textint_map.pred_to_input(preds)
        return inputs

    # TODO: no more price buffer
    def run_decode(self, sess, max_len, batch_size=1, stop_symbol=None, **kwargs):
        #return self.decoder.run_decode(sess, max_len, batch_size=batch_size, stop_symbol=stop_symbol, **kwargs)
        if stop_symbol is not None:
            assert batch_size == 1, 'Early stop only works for single instance'
        price_symbol = kwargs.pop('price_symbol')
        feed_dict = self.get_feed_dict(**kwargs)
        preds = np.zeros([batch_size, max_len], dtype=np.int32)
        prices = np.zeros([batch_size, max_len], dtype=np.float32)
        # reshape: squeeze step dim; we are only considering one step
        price_buffer = PriceBuffer(init_price_batch=kwargs['price_predictor']['inputs'].reshape(batch_size, -1))

        for i in xrange(max_len):
            logits, final_state, price = sess.run((self.output_dict['logits'], self.output_dict['final_state'], self.output_dict['prices']), feed_dict=feed_dict)
            step_preds = self.decoder.sampler.sample(logits)  # (batch_size, 1)

            preds[:, [i]] = step_preds
            if step_preds[0][0] == stop_symbol:
                break

            # Update price
            mask = (step_preds == price_symbol).reshape(-1)
            # At least one <price>
            if np.sum(mask) > 0:
                # NOTE: price is (batch_size, 1)
                price_buffer.add(price.reshape(batch_size), mask, True)
            prices[:, [i]] = price

            price_batch = price_buffer.to_price_batch()
            #print price_batch
            feed_dict = self.get_feed_dict(inputs=self.pred_to_input(step_preds, kwargs['textint_map']),
                    price_predictor={'inputs': price_batch},
                    init_state=final_state)

        # TODO: hack
        #print prices
        prices = np.around(prices, decimals=2)
        return {'preds': preds, 'prices': prices, 'final_state': final_state}

class ClassifyDecoder(DecoderWrapper):
    def __init__(self, decoder):
        self.decoder = decoder
        self.prompt_len = decoder.prompt_len
        self.pad = self.decoder.pad
        self.feedable_vars = {}

    def get_feed_dict(self, **kwargs):
        feed_dict = super(ClassifyDecoder, self).get_feed_dict(**kwargs)
        feed_dict[self.feedable_vars['candidates']] = kwargs.pop('candidates')
        optional_add(feed_dict, self.feedable_vars['labels'], kwargs.pop('candidate_labels', None))
        return feed_dict

    def _tile_inputs(self, input_dict, multiplier):
        with tf.name_scope('tile_inputs'):
            if hasattr(self.decoder, 'context_embedding'):
                self.decoder.context_embedding = tile_tensor(self.decoder.context_embedding, multiplier)
            for input_name, tensor in input_dict.iteritems():
                if input_name == 'init_cell_state':
                    input_dict[input_name] = tile_tensor(tensor, multiplier)
                elif input_name == 'encoder_embeddings':  # (seq_len, batch_size, embed_size)
                    input_dict[input_name] = transpose_first_two_dims(
                            tile_tensor(transpose_first_two_dims(tensor), multiplier))
                elif input_name == 'inputs':
                    continue
                elif input_name == 'price_history':
                    # TODO
                    continue
                else:
                    print '{} not tiled'.format(input_name)
                    raise ValueError
        return input_dict

    def build_model(self, input_dict):
        with tf.variable_scope(type(self).__name__):
            # Inputs
            candidates = tf.placeholder(tf.int32, shape=[None, None, None], name='candidates')  # (batch_size, num_candidates, seq_len)
            labels = tf.placeholder(tf.int32, shape=[None, None], name='candidate_labels')  # Binary label: (batch_size, num_candidates)
            self.feedable_vars['candidates'] = candidates
            self.feedable_vars['labels'] = labels
            shape = tf.shape(candidates)
            batch_size, num_candidates = shape[0], shape[1]

            input_dict = self._tile_inputs(input_dict, num_candidates)
            input_dict['inputs'] = tf.reshape(candidates, [batch_size * num_candidates, -1])
            self.decoder.build_model(input_dict)
            output = self.decoder.output_dict['final_output']
            # NOTE: need to preserve the static shape
            embed_size = output.shape.as_list()[-1]
            candidate_embeddings = tf.reshape(output, [batch_size, num_candidates, embed_size])

            scores = tf.squeeze(tf.layers.dense(candidate_embeddings, 1, activation=None, use_bias=False), axis=2)  # (batch_size, num_candidates, 1)
            self.output_dict = {
                    'candidate_embeddings': candidate_embeddings,
                    'scores': scores,
                    }

    def compute_loss(self):
        # Check padded candidates by checking if the first token is PAD
        candidate_first_tokens = self.feedable_vars['candidates'][:, :, 0]  # (batch_size, num_candidates)
        non_padding_candidate = tf.not_equal(candidate_first_tokens, tf.constant(self.pad))
        labels = self.feedable_vars['labels']  # (batch_size, num_candidates)
        positive_candidates = tf.equal(labels, tf.constant(1))
        negative_candidates = tf.equal(labels, tf.constant(0))
        positive_mask = tf.logical_and(positive_candidates, non_padding_candidate)
        negative_mask = tf.logical_and(negative_candidates, non_padding_candidate)

        # Make the 3D mask
        num_candidates = tf.shape(labels)[1]
        # For each positive example score, mark all negative example scores
        positive_mask = tf.tile(tf.expand_dims(positive_mask, 2), [1, 1, num_candidates])
        negative_mask = tf.tile(tf.expand_dims(negative_mask, 1), [1, num_candidates, 1])
        mask = tf.logical_and(positive_mask, negative_mask)

        # For each positive example, compute the diff
        scores = self.output_dict['scores']  # (batch_size, num_candidates)
        positive_score_matrix = tf.tile(tf.expand_dims(scores, 2), [1, 1, num_candidates])
        positive_scores = tf.where(mask, positive_score_matrix, tf.ones_like(positive_score_matrix) * 1.1)
        negative_score_matrix = tf.tile(tf.expand_dims(scores, 1), [1, num_candidates, 1])
        negative_scores = tf.where(mask, negative_score_matrix, tf.zeros_like(negative_score_matrix))

        num_pairs = tf.reduce_sum(tf.cast(mask, tf.float32))
        total_loss = tf.reduce_sum(tf.maximum(1. + negative_scores - positive_scores, 0))
        loss = total_loss / (num_pairs + EPS)

        # Statistics
        tf.summary.scalar('candidate_ranking_loss', loss)

        return loss, (total_loss, num_pairs)
