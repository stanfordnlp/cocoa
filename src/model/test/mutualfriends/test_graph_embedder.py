import pytest
import tensorflow as tf
import numpy as np
from numpy.testing import assert_array_equal
from model.graph_embedder import GraphEmbedder, GraphEmbedderConfig

class TestGraphEmbedder(object):
    num_nodes = 5

    def test_update_utterance(self, graph_embedder):
        config = graph_embedder.config
        init_utterances = tf.zeros([2, self.num_nodes, config.utterance_size])
        entity_indices = tf.constant([[1, 2], [3, 4]])
        utterance = tf.placeholder(tf.float32, shape=[None, None])
        numpy_utterance = np.array([[1,1,1,1],[2,2,2,2]])
        updated_utterances = graph_embedder.update_utterance(entity_indices, utterance, init_utterances)
        with tf.Session() as sess:
            [ans] = sess.run([updated_utterances], feed_dict={utterance: numpy_utterance})

        expected_ans = np.array([[[0,0,0,0],\
                                  [1,1,1,1],\
                                  [1,1,1,1],\
                                  [0,0,0,0],\
                                  [0,0,0,0],\
                                 ],\
                                 [[0,0,0,0],\
                                  [0,0,0,0],\
                                  [0,0,0,0],\
                                  [2,2,2,2],\
                                  [2,2,2,2],\
                                 ]])
        assert_array_equal(ans, expected_ans)

    def test_embed_path(self, graph_embedder, capsys):
        node_embedding = tf.constant([[[0,0,0,0],
                                       [1,1,1,1],
                                       [2,2,2,2]],
                                      [[0,0,0,0],
                                       [2,2,2,2],
                                       [1,1,1,1]]], dtype=tf.float32)
        edge_embedding = tf.constant([[0,0,0,0]], dtype=tf.float32)

        path_pad = (0, 0, 0)
        paths = tf.constant([[[0,0,1], [0,0,2], path_pad],
                             [[1,0,2], path_pad, path_pad]], dtype=tf.int32)
        path_embeds = graph_embedder.embed_path(node_embedding, edge_embedding, paths)

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            [ans] = sess.run([path_embeds])

        assert_array_equal(ans[0][-1], ans[1][1])
        assert_array_equal(ans[0][0], ans[1][0])

    def test_pass_message(self, graph_embedder):
        pad = 2
        path_embeds = tf.constant([[[0,0,0],[1,1,1],[-1,-1,-1]],
                                   [[2,2,2],[-1,-1,-1],[-1,-1,-1]]], dtype=tf.float32)
        neighbors = tf.constant([[[0,1], [pad,pad], [pad,pad], [pad,pad]],
                                 [[pad,pad], [0,pad], [pad,pad], [pad,pad]]], dtype=tf.int32)
        new_embed = graph_embedder.pass_message(path_embeds, neighbors, pad)

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            [ans] = sess.run([new_embed])

        expected_ans = np.array([[[1,1,1],[0,0,0],[0,0,0],[0,0,0]],
                                 [[0,0,0],[2,2,2],[0,0,0],[0,0,0]]])
        assert_array_equal(ans, expected_ans)

    def test_get_context(self, graph_embedder, capsys):
        config = graph_embedder.config
        node_ids = np.array([[0,1,2], [0,1,2]], dtype=np.int32)
        entity_ids = np.zeros([2, 3], dtype=np.int32)
        pad_path = (0, 0, 0)
        paths = np.array([[pad_path, [0,0,1], [0,0,2]],\
                          [pad_path, [1,0,2], pad_path]], dtype=np.int32)
        node_paths = np.array([[[1,2], [0,0], [0,0]],\
                               [[0,0], [1,0], [0,0]]], dtype=np.int32)
        node_feats = np.ones([2, 3, config.feat_size], dtype=np.float)
        utterances = tf.constant(np.zeros([2, self.num_nodes, config.utterance_size], dtype=np.float), dtype=tf.float32)
        input_data = (node_ids, entity_ids, paths, node_paths, node_feats)
        feed_dict = {graph_embedder.input_data: input_data}

        context2, mask2 = graph_embedder.get_context(utterances)
        config.mp_iters = 1
        tf.get_variable_scope().reuse_variables()
        context1, mask1 = graph_embedder.get_context(utterances)

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            [ans1, ans2, m1, m2] = sess.run([context1, context2, mask1, mask2], feed_dict = feed_dict)
        expected_mask = np.array([[True, False, False], [False, True, False]])
        assert_array_equal(m1, expected_mask)
        assert_array_equal(m2, expected_mask)
        with capsys.disabled():
            print 'Before update:'
            print utterances
            print 'After update once:'
            print ans1.shape
            print ans1
            print 'After update twice:'
            print ans2.shape
            print ans2
        if config.message_combiner == 'concat':
            assert_array_equal(ans1, ans2[:,:,:8])
