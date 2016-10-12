import pytest
import numpy as np
import tensorflow as tf
from numpy.testing import assert_array_equal
from model.util import batch_embedding_lookup, batch_linear

class TestUtil(object):
    def test_batch_embedding_lookup(self):
        embeddings = tf.constant([[[0,0,0],[1,1,1],[2,2,2]],
                                  [[0,0,0],[3,3,3],[4,4,4]]])
        indices = tf.constant([[0,1],[1,2]])
        embeds = batch_embedding_lookup(embeddings, indices)

        with tf.Session() as sess:
            [ans] = sess.run([embeds])
        expected_ans = np.array([[[0,0,0],[1,1,1]],
                                 [[3,3,3],[4,4,4]]])
        assert_array_equal(ans, expected_ans)

    def test_batch_linear(self):
        a = tf.ones([2,3,4])
        b = tf.zeros([2,3,2])
        c = batch_linear([a, b], 3, True)
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            [ans] = sess.run([c])
        assert ans.shape == (2, 3, 3)
        assert_array_equal(ans[0], ans[1])

