import tensorflow as tf
import numpy as np
from src.model.util import transpose_first_two_dims, batch_linear, batch_embedding_lookup, EPS
from src.model.encdec import BasicEncoder, BasicDecoder, Sampler, optional_add
from preprocess import markers

class BasicEncoderDecoder(object):
    '''
    Basic seq2seq model.
    '''
    def __init__(self, encoder_word_embedder, decoder_word_embedder, encoder, decoder, pad, select, re_encode=False, scope=None):
        self.PAD = pad  # Id of PAD in the vocab
        self.encoder = encoder
        self.decoder = decoder
        #self.re_encode = re_encode
        self.tf_variables = set()
        self.build_model(encoder_word_embedder, decoder_word_embedder, encoder, decoder, scope)

    def compute_loss(self, output_dict, targets):
        return self.decoder.compute_loss(targets, self.PAD, self.SELECT)

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
                encoder.build_model(encoder_word_embedder, encoder_input_dict, self.tf_variables, time_major=False)

            # Decoding
            with tf.name_scope('Decoder'):
                decoder_input_dict = self._decoder_input_dict(encoder.output_dict)
                decoder.build_model(decoder_word_embedder, decoder_input_dict, self.tf_variables, time_major=False)

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

            self.targets = tf.placeholder(tf.int32, shape=[None, None], name='targets')

            # Loss
            self.loss, self.seq_loss, self.total_loss, self.select_loss = self.compute_loss(decoder.output_dict, self.targets)

    def get_feed_dict(self, **kwargs):
        feed_dict = kwargs.pop('feed_dict', {})
        feed_dict = self.encoder.get_feed_dict(**kwargs.pop('encoder'))
        feed_dict = self.decoder.get_feed_dict(feed_dict=feed_dict, **kwargs.pop('decoder'))
        optional_add(feed_dict, self.targets, kwargs.pop('targets', None))
        return feed_dict

    def generate(self, sess, batch, encoder_init_state, max_len, copy=False, vocab=None, graphs=None, utterances=None, textint_map=None):
        encoder_inputs = batch['encoder_inputs']
        decoder_inputs = batch['decoder_inputs']
        batch_size = encoder_inputs.shape[0]

        # Encode true prefix
        # TODO: get_encoder_args
        encoder_args = {'inputs': encoder_inputs,
                'last_inds': batch['encoder_inputs_last_inds'],
                'init_state': encoder_init_state
                }
        encoder_output_dict = self.encoder.run_encode(sess, **encoder_args)

        # Decode max_len steps
        # TODO: get_decoder_args
        decoder_args = {'inputs': decoder_inputs[:, [0]],
                'last_inds': np.zeros([batch_size], dtype=np.int32),
                'init_state': encoder_output_dict['final_state'],
                'textint_map': textint_map
                }
        decoder_output_dict = self.decoder.run_decode(sess, max_len, batch_size, **decoder_args)

        # Decode true utterances (so that we always condition on true prefix)
        decoder_args['inputs'] = decoder_inputs
        decoder_args['last_inds'] = batch['decoder_inputs_last_inds']
        feed_dict = self.decoder.get_feed_dict(**decoder_args)
        # TODO: this is needed by re-encode
        #feed_dict[self.encoder.keep_prob] = 1. - self.encoder.dropout
        true_final_state = sess.run((self.final_state), feed_dict=feed_dict)
        return {'preds': decoder_output_dict['preds'],
                'final_state': decoder_output_dict['final_state'],
                'true_final_state': true_final_state,
                }

