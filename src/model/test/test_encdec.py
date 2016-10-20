import pytest
import tensorflow as tf
import numpy as np
from numpy.testing import assert_array_equal
from model.encdec import BasicEncoder, BasicDecoder, BasicEncoderDecoder, GraphEncoder, GraphDecoder, GraphEncoderDecoder
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
    def basic_input_dict(self):
        input_dict = {'inputs': tf.constant(np.random.randint(self.vocab_size, size=(self.batch_size, self.seq_len)), dtype=tf.int32)}
        return input_dict

    @pytest.fixture(scope='session')
    def last_inds(self):
        return np.full([self.batch_size], 2, dtype=np.int32)

    def test_basic_encoder(self, basic_encoder, word_embedder, basic_input_dict, last_inds):
        output_dict = basic_encoder.build_model(basic_input_dict, word_embedder, time_major=False)
        outputs, final_state = output_dict['outputs'], output_dict['final_state']
        feed_dict = {basic_encoder.last_inds: last_inds}

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            [outputs, final_state] = sess.run([outputs, final_state], feed_dict=feed_dict)

        assert outputs.shape == (self.seq_len, self.batch_size, self.rnn_size)

    @pytest.fixture(scope='session')
    def basic_decoder(self):
        return BasicDecoder(self.rnn_size, self.num_symbols, batch_size=self.batch_size)

    def test_basic_decoder(self, basic_decoder, word_embedder, basic_input_dict, last_inds):
        output_dict = basic_decoder.build_model(basic_input_dict, word_embedder, time_major=False)
        logits, final_state = output_dict['logits'], output_dict['final_state']
        feed_dict = {basic_decoder.last_inds: last_inds}

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            [logits] = sess.run([logits], feed_dict=feed_dict)

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

    @pytest.fixture(scope='session')
    def graph_encoder(self, graph_embedder):
        return GraphEncoder(self.rnn_size, graph_embedder, batch_size=self.batch_size)

    @pytest.fixture(scope='session')
    def graph_decoder(self, graph_embedder):
        return GraphDecoder(self.rnn_size, self.num_symbols, graph_embedder, batch_size=self.batch_size)

    @pytest.fixture(scope='session')
    def graph_encoderdecoder(self, word_embedder, graph_embedder, graph_encoder, graph_decoder):
        return GraphEncoderDecoder(word_embedder, graph_embedder, graph_encoder, graph_decoder, self.pad)

    def test_graph_encoderdecoder(self, graph_encoderdecoder, graph_batch, tokens, word_seq, last_inds):
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            graph_data = graph_batch.get_batch_data(tokens, tokens, None)
            feed_dict = graph_encoderdecoder.update_feed_dict(encoder_inputs=word_seq, decoder_inputs=word_seq, targets=word_seq, encoder_inputs_last_inds=last_inds, decoder_inputs_last_inds=last_inds)
            feed_dict = graph_encoderdecoder.update_feed_dict(feed_dict=feed_dict,
                    encoder_entities=graph_data['encoder_entities'],
                    decoder_entities=graph_data['decoder_entities'],
                    encoder_input_utterances=graph_data['utterances'],
                    graph_structure=(graph_data['node_ids'], graph_data['entity_ids'], graph_data['paths'], graph_data['node_paths'], graph_data['node_feats']))
            [decoder_final_state, loss] = sess.run([graph_encoderdecoder.decoder_final_state, graph_encoderdecoder.loss], feed_dict=feed_dict)

    def test_basic_encoderdecoder(self, basic_encoderdecoder, basic_encoder, word_seq, last_inds):
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            feed_dict = basic_encoderdecoder.update_feed_dict(encoder_inputs=word_seq, decoder_inputs=word_seq, targets=word_seq, encoder_inputs_last_inds=last_inds, decoder_inputs_last_inds=last_inds)
            [decoder_final_state, loss] = sess.run([basic_encoderdecoder.decoder_final_state, basic_encoderdecoder.loss], feed_dict=feed_dict)

            # Test given encoder_init_state
            feed_dict = basic_encoderdecoder.update_feed_dict(encoder_init_state=decoder_final_state, encoder_inputs=word_seq)
            [encoder_init_state] = sess.run([basic_encoder.init_state], feed_dict=feed_dict)
            assert_array_equal(encoder_init_state, decoder_final_state)

    def test_loss_mask(self, basic_encoderdecoder, capsys):
        logits = tf.constant(np.ones([2,4,5]), dtype=tf.float32)
        targets = tf.constant(np.ones([2,4]) * self.pad, dtype=tf.int32)
        loss = basic_encoderdecoder.compute_loss(logits, targets)
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            [loss] = sess.run([loss])
        assert loss == 0

    @pytest.fixture(scope='session')
    def batch(self, word_seq, last_inds, tokens):
        return {'encoder_inputs': word_seq,
                'decoder_inputs': word_seq,
                'encoder_tokens': tokens,
                'decoder_tokens': tokens,
                'encoder_inputs_last_inds': last_inds,
                'decoder_inputs_last_inds': last_inds,
               }

    def test_basic_generate(self, basic_encoderdecoder, batch, capsys):
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            preds, final_state, true_final_state, _ = basic_encoderdecoder.generate(sess, batch, None, 10)
        assert preds.shape == (2, 10)
        with capsys.disabled():
            print '\npreds:\n', preds
            print 'final state:\n', final_state
            print 'true final state:\n', true_final_state

    @pytest.mark.only
    def test_graph_generate(self, batch, graph_batch, graph_encoderdecoder, capsys):
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            preds, final_state, true_final_state, utterances = graph_encoderdecoder.generate(sess, batch, None, 10, graphs=graph_batch)
        with capsys.disabled():
            print 'utterances:', utterances

