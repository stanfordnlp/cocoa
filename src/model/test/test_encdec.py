import pytest
import tensorflow as tf
import numpy as np
from numpy.testing import assert_array_equal
from model.encdec import BasicEncoder, BasicDecoder, BasicEncoderDecoder
from model.word_embedder import WordEmbedder

class TestEncoderDecoder(object):
    rnn_size = 3
    batch_size = 2
    seq_len = 4
    input_size = 5
    num_symbols = 5
    vocab_size = 5
    word_embed_size = 4
    pad = 0

    @pytest.fixture(scope='session')
    def basic_encoder(self):
        return BasicEncoder(self.rnn_size, batch_size=self.batch_size)

    @pytest.fixture(scope='session')
    def seq_inputs(self):
        inputs = tf.random_uniform(shape=[self.batch_size, self.seq_len, self.input_size])
        return inputs

    def test_basic_encoder(self, basic_encoder, seq_inputs):
        outputs, states = basic_encoder.build_model(seq_inputs, time_major=False)

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            [outputs, states] = sess.run([outputs, states])

        assert outputs.shape == (self.seq_len, self.batch_size, self.rnn_size)

    @pytest.fixture(scope='session')
    def basic_decoder(self):
        return BasicDecoder(self.rnn_size, self.num_symbols, batch_size=self.batch_size)

    def test_basic_decoder(self, basic_decoder, seq_inputs):
        logits, final_state = basic_decoder.build_model(seq_inputs, time_major=False)

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            [logits] = sess.run([logits])

        assert logits.shape == (self.batch_size, self.seq_len, self.num_symbols)

    @pytest.fixture(scope='session')
    def word_embedder(self):
        return WordEmbedder(self.vocab_size, self.word_embed_size)

    @pytest.fixture(scope='session')
    def basic_encoderdecoder(self, word_embedder, basic_encoder, basic_decoder):
        return BasicEncoderDecoder(word_embedder, basic_encoder, basic_decoder, self.pad)

    @pytest.fixture(scope='session')
    def word_seq(self):
        return np.array([[0,1,2],
                         [2,1,3]])

    def test_basic_encoderdecoder(self, basic_encoderdecoder, word_seq):
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            feed_dict = basic_encoderdecoder.update_feed_dict(encoder_inputs=word_seq, decoder_inputs=word_seq, targets=word_seq)
            [decoder_final_state, loss] = sess.run([basic_encoderdecoder.decoder_final_state, basic_encoderdecoder.loss], feed_dict=feed_dict)

            # Test give decoder_init_state
            feed_dict = basic_encoderdecoder.update_feed_dict(decoder_init_state=decoder_final_state, decoder_inputs=word_seq, targets=word_seq)
            [decoder_final_state, loss] = sess.run([basic_encoderdecoder.decoder_final_state, basic_encoderdecoder.loss], feed_dict=feed_dict)

    def test_loss_mask(self, basic_encoderdecoder, capsys):
        logits = tf.constant(np.ones([2,4,5]), dtype=tf.float32)
        targets = tf.constant(np.ones([2,4]) * self.pad, dtype=tf.int32)
        loss = basic_encoderdecoder.compute_loss(logits, targets)
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            [loss] = sess.run([loss])
        assert loss == 0
