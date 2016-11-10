import pytest
from model.rnn_cell import AttnRNNCell
import numpy as np
import tensorflow as tf

class TestAttnRNNCell(object):
    rnn_size = 2
    context_size = 3
    batch_size = 2
    context_len = 4

    @pytest.fixture(scope='session')
    def cell(self):
        return AttnRNNCell(self.rnn_size, self.context_size)

    @pytest.fixture(scope='session')
    def query(self):
        return tf.constant(np.random.rand(self.batch_size, self.rnn_size), dtype=tf.float32)

    @pytest.fixture(scope='session')
    def context(self):
        context = tf.constant(np.random.rand(self.batch_size, self.context_len, self.context_size), dtype=tf.float32)
        np_mask = np.array([[True, False, True, False],
                            [True, False, False, True]])
        mask = tf.constant(np_mask, dtype=tf.bool)
        return context, mask, np_mask

    def test_compute_attention(self, cell, query, context):
        context, mask, np_mask = context
        weighted_context, attn_scores = cell.compute_attention(query, context, mask)
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            [attn_scores] = sess.run([attn_scores])
        assert np.all(np.isneginf(attn_scores[np.invert(np_mask)]))
