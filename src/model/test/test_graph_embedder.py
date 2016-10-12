import pytest
import tensorflow as tf
import numpy as np
from numpy.testing import assert_array_equal
from model.graph_embedder import GraphEmbedder, GraphEmbedderConfig

class TestGraphEmbedder(object):
    @pytest.fixture(scope='session')
    def config(self):
        num_nodes = 5
        num_edge_labels = 1
        node_embed_size = 4
        edge_embed_size = 4
        utterance_size = 4
        feat_size = 1
        batch_size = 2
        max_degree = 2
        return GraphEmbedderConfig(num_nodes, num_edge_labels, node_embed_size, edge_embed_size, utterance_size, feat_size, batch_size=batch_size, max_degree=max_degree)

    @pytest.fixture(scope='session')
    def graph_embedder(self, config):
        return GraphEmbedder(config)

    def test_update_utterance(self, graph_embedder):
        config = graph_embedder.config
        init_utterances = tf.zeros([2, config.num_nodes, config.utterance_size])
        entity_indices = tf.constant([[1, 2], [3, 4]])
        utterance = tf.concat(0, [tf.ones([1, config.utterance_size]), tf.ones([1, config.utterance_size])*2])
        updated_utterances = graph_embedder.update_utterance(entity_indices, utterance, init_utterances)
        with tf.Session() as sess:
            [ans] = sess.run([updated_utterances])

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

        path_pad = graph_embedder.config.PATH_PAD
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
        pad_path = config.PATH_PAD
        paths = np.array([[pad_path, [0,0,1], [0,0,2]],\
                          [pad_path, [1,0,2], pad_path]], dtype=np.int32)
        node_paths = np.array([[[1,2], [0,0], [0,0]],\
                               [[0,0], [1,0], [0,0]]], dtype=np.int32)
        node_feats = np.ones([2, 3, config.feat_size], dtype=np.float)
        utterances = np.zeros([2, config.num_nodes, config.utterance_size], dtype=np.float)
        input_data = (node_ids, entity_ids, paths, node_paths, node_feats, utterances)
        feed_dict = {graph_embedder.input_data: input_data}

        context2 = graph_embedder.get_context()
        config.mp_iters = 1
        tf.get_variable_scope().reuse_variables()
        context1 = graph_embedder.get_context()

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            [ans1, ans2] = sess.run([context1, context2], feed_dict = feed_dict)
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
