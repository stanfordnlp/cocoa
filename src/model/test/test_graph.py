import pytest
from model.graph import Graph, GraphMetadata, GraphBatch
from basic.schema import Schema
from model.preprocess import build_schema_mappings
from basic.kb import KB
import numpy as np
from numpy.testing import assert_array_equal

class TestGraph(object):
    def test_mappings(self, graph, capsys):
        with capsys.disabled():
            print 'Entity map:'
            graph.metadata.entity_map.dump()
            print 'Relation map:'
            graph.metadata.relation_map.dump()

    def test_nodes(self, graph, capsys):
        with capsys.disabled():
            print '%d nodes in graph:' % graph.nodes.size
            graph.nodes.dump()
        assert graph.nodes.size == 10

    def test_paths(self, graph, capsys):
        def print_path(i, path):
            n1, r, n2 = path
            print i, \
                  graph.nodes.to_word(n1), \
                  graph.metadata.relation_map.to_word(r), \
                  graph.nodes.to_word(n2)
        with capsys.disabled():
            print '%d paths in graph:' % len(graph.paths)
            for i, path in enumerate(graph.paths):
                print_path(i, path)
        assert graph.paths.shape[0] == 11 * 2 + 1

    def test_node_paths(self, graph, capsys):
        assert len(graph.node_paths) == graph.nodes.size
        with capsys.disabled():
            print 'Node paths:'
            for i, node_path in enumerate(graph.node_paths):
                print graph.nodes.to_word(graph.node_ids[i]), node_path

    def test_features(self, graph, capsys):
        with capsys.disabled():
            print 'Node features:'
            features = graph.get_features()
            print 'degrees:'
            print features[:, :4]
            print 'node types:'
            print features[:, 4:]

    def test_add_entity(self, graph, capsys):
        graph.add_entity_nodes([('facebook', 'company')])
        assert graph.nodes.size == 11
        assert graph.nodes.to_ind(('facebook', 'company')) == 10
        # Features are incrementally updated
        assert_array_equal(graph.feats, graph.get_features())
        assert graph.feats.shape[0] == graph.nodes.size
        assert_array_equal(graph.node_paths[10], np.array([0]))

    def test_read_utterance(self, graph, capsys):
        graph.read_utterance([('alice', ('alice', 'person')), 'works', 'at', ('google', ('google', 'company'))])
        alice = graph.nodes.to_ind(('alice', 'person'))
        google = graph.nodes.to_ind(('google', 'company'))
        assert_array_equal(graph.entities, [alice, -1, -1, google])
        assert graph.nodes.size == graph.feats.shape[0]
        assert_array_equal(graph.get_entity_list(2), [[alice], [alice, google]])

    def test_update_utterances(self, graph_batch):
        utterances = np.ones([2, 3, Graph.metadata.utterance_size])
        max_num_nodes = 5
        utterances = graph_batch.update_utterances(utterances, max_num_nodes)

        x = np.zeros([max_num_nodes+1, Graph.metadata.utterance_size])
        x[:3, :] = 1
        expected = np.tile(np.expand_dims(x, 0), [2, 1, 1])
        assert_array_equal(utterances, expected)

    def test_batch_entity_lists(self, graph_batch):
        entity_lists = [[1,2,3], [4]]
        batch_entity_lists = graph_batch._batch_entity_lists(entity_lists, 5)
        expected = np.array([[2,3], [4,5]])
        assert_array_equal(batch_entity_lists, expected)

    def test_make_batch(self, graph_batch, capsys):
        encoder_tokens = None
        u1 = [('alice', ('alice', 'person')), 'works', 'at', ('google', ('google', 'company'))]
        u2 = [('ian', ('ian', 'person')), 'likes', ('hiking', ('hiking', 'hobby'))]
        decoder_tokens = [u1, u2]
        with capsys.disabled():
            batch = graph_batch.get_batch_data(encoder_tokens, decoder_tokens)

        max_num_nodes = graph_batch._max_num_nodes()
        assert max_num_nodes == 12
        assert batch['utterances'].shape[1] == max_num_nodes + 1

        with capsys.disabled():
            for k, v in batch.iteritems():
                print k
                print v
