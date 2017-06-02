import tensorflow as tf
import numpy as np
from src.model.util import transpose_first_two_dims, batch_linear, batch_embedding_lookup, EPS
from src.model.encdec import BasicEncoder, BasicDecoder, Sampler, optional_add
from preprocess import markers, START_PRICE
from price_buffer import PriceBuffer

# TODO: refactor this class
class BasicEncoderDecoder(object):
    '''
    Basic seq2seq model.
    '''
    def __init__(self, encoder_word_embedder, decoder_word_embedder, encoder, decoder, pad, re_encode=False, scope=None):
        self.PAD = pad  # Id of PAD in the vocab
        self.encoder = encoder
        self.decoder = decoder
        #self.re_encode = re_encode
        self.tf_variables = set()
        self.build_model(encoder_word_embedder, decoder_word_embedder, encoder, decoder, scope)

    def compute_loss(self, output_dict):
        return self.decoder.compute_loss(self.PAD)

    def _encoder_input_dict(self):
        return {
                'init_state': None,
               }

    def _decoder_input_dict(self, encoder_output_dict):
        return {
                'init_state': encoder_output_dict['final_state'],
               }

    def build_model(self, encoder_word_embedder, decoder_word_embedder, encoder, decoder, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # Encoding
            with tf.name_scope('Encoder'):
                encoder_input_dict = self._encoder_input_dict()
                encoder.build_model(encoder_word_embedder, encoder_input_dict, self.tf_variables, time_major=False, pad=self.PAD)

            # Decoding
            with tf.name_scope('Decoder'):
                decoder_input_dict = self._decoder_input_dict(encoder.output_dict)
                decoder.build_model(decoder_word_embedder, decoder_input_dict, self.tf_variables, time_major=False, pad=self.PAD)

            # Re-encode decoded sequence
            # TODO: re-encode is not implemeted in neural_sessions yet
            # TODO: hierarchical
            #if self.re_encode:
            #    input_args = decoder.get_rnn_inputs_args()
            #    input_args['init_state'] = encoder.output_dict['final_state']
            #    reencode_output_dict = encoder.encode(time_major=False, **input_args)
            #    self.final_state = reencode_output_dict['final_state']
            #else:
            self.final_state = decoder.get_encoder_state(decoder.output_dict['final_state'])

            #self.targets = tf.placeholder(tf.int32, shape=[None, None], name='targets')

            # Loss
            self.loss, self.seq_loss, self.total_loss = self.compute_loss(decoder.output_dict)

    def get_feed_dict(self, **kwargs):
        feed_dict = kwargs.pop('feed_dict', {})
        feed_dict = self.encoder.get_feed_dict(**kwargs.pop('encoder'))
        feed_dict = self.decoder.get_feed_dict(feed_dict=feed_dict, **kwargs.pop('decoder'))
        return feed_dict

    def generate(self, sess, batch, encoder_init_state, max_len, textint_map=None):
        encoder_inputs = batch['encoder_inputs']
        decoder_inputs = batch['decoder_inputs']
        batch_size = encoder_inputs.shape[0]

        # Encode true prefix
        # TODO: get_encoder_args
        encoder_args = {'inputs': encoder_inputs,
                'init_state': encoder_init_state
                }
        encoder_output_dict = self.encoder.run_encode(sess, **encoder_args)

        # Decode max_len steps
        # TODO: get_decoder_args for refactor
        price_args = {'inputs': batch['decoder_price_inputs'][:, [0]]}
        decoder_args = {'inputs': decoder_inputs[:, [0]],
                'init_state': encoder_output_dict['final_state'],
                'price_symbol': textint_map.vocab.to_ind('<price>'),
                'price_predictor': price_args,
                'textint_map': textint_map
                }
        decoder_output_dict = self.decoder.run_decode(sess, max_len, batch_size, **decoder_args)

        # Decode true utterances (so that we always condition on true prefix)
        decoder_args['inputs'] = decoder_inputs
        feed_dict = self.decoder.get_feed_dict(**decoder_args)
        # TODO: this is needed by re-encode
        #feed_dict[self.encoder.keep_prob] = 1. - self.encoder.dropout
        true_final_state = sess.run((self.final_state), feed_dict=feed_dict)
        return {'preds': decoder_output_dict['preds'],
                'prices': decoder_output_dict.get('prices', None),
                'final_state': decoder_output_dict['final_state'],
                'true_final_state': true_final_state,
                }
        #return {'preds': decoder_output_dict['preds'],
        #        'prices': None,
        #        'final_state': decoder_output_dict['final_state'],
        #        'true_final_state': true_final_state,
        #        }

class PriceDecoder(object):
    '''
    A wrapper of a decoder that outputs <price> and a price predictor that fills in the actual price.
    '''
    def __init__(self, decoder, price_predictor):
        self.decoder = decoder
        self.price_predictor = price_predictor

    def build_model(self, word_embedder, input_dict, tf_variables, pad=0, time_major=True, scope=None):
        self.decoder.build_model(word_embedder, input_dict, tf_variables, pad=pad, time_major=time_major, scope=scope)
        # NOTE: output from rnn is time major
        context = transpose_first_two_dims(self.decoder.output_dict['outputs'])
        self.price_predictor.build_model(context)

        # Outputs
        self.output_dict = dict(self.decoder.output_dict)
        self.output_dict['prices'] = self.price_predictor.output_dict['prices']

    def compute_loss(self, pad):
        loss, seq_loss, total_loss = self.decoder.compute_loss(pad)
        price_loss = self.price_predictor.compute_loss(pad)
        loss += price_loss
        # NOTE: seq_loss and total_loss do not depend on price_loss. We're using loss for bp.
        return loss, seq_loss, total_loss

    def get_feed_dict(self, **kwargs):
        feed_dict = self.decoder.get_feed_dict(**kwargs)
        feed_dict = self.price_predictor.get_feed_dict(feed_dict=feed_dict, **kwargs.pop('price_predictor'))
        return feed_dict

    def get_encoder_state(self, state):
        '''
        Given the hidden state to the encoder to continue from there.
        '''
        return self.decoder.get_encoder_state(state)

    def pred_to_input(self, preds, **kwargs):
        '''
        Convert predictions to input of the next decoding step.
        '''
        textint_map = kwargs.pop('textint_map')
        inputs = textint_map.pred_to_input(preds)
        return inputs

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
            feed_dict = self.get_feed_dict(inputs=self.pred_to_input(step_preds, **kwargs),
                    price_predictor={'inputs': price_batch},
                    init_state=final_state)

        # TODO: hack
        #print prices
        prices = np.around(prices, decimals=2)
        return {'preds': preds, 'prices': prices, 'final_state': final_state}
