import pytest
from model.graph import Graph, GraphMetadata, GraphBatch
from basic.schema import Schema
from basic.kb import KB
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from model.vocab import Vocabulary
from model.preprocess import TextIntMap, Preprocessor

@pytest.fixture(scope='session')
def preprocessor(schema, lexicon):
    return Preprocessor(schema, lexicon, 'canonical', 'canonical', 'graph')

@pytest.fixture(scope='session')
def vocab():
    vocab = Vocabulary(unk=False)
    vocab.add_words(['work', 'like', ('alice', 'person'), ('hiking', 'hobby')])
    return vocab

@pytest.fixture(scope='session')
def textint_map(vocab, metadata, preprocessor):
    return TextIntMap(vocab, metadata.entity_map, preprocessor)

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

        x = np.zeros([Graph.metadata.max_num_entities+1, Graph.metadata.utterance_size])
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
            batch = graph_batch.get_batch_data(encoder_tokens, decoder_tokens, None)

        max_num_nodes = graph_batch._max_num_nodes()
        assert max_num_nodes == 12
        assert batch['utterances'].shape[1] >= max_num_nodes + 1

        with capsys.disabled():
            for k, v in batch.iteritems():
                print k
                print v

    def test_copy(self, graph_batch, metadata, preprocessor, vocab, textint_map):
        alice = metadata.entity_map.to_ind(('alice', 'person')) + vocab.size
        hiking = metadata.entity_map.to_ind(('hiking', 'hobby')) + vocab.size
        targets = np.array([[0, alice, hiking],
                            [1, alice, hiking]])
        new_targets = graph_batch.copy_targets(targets, vocab.size)
        assert graph_batch.graphs[0].nodes.to_word(new_targets[0][1]-vocab.size) == ('alice', 'person')
        assert graph_batch.graphs[0].nodes.to_word(new_targets[0][2]-vocab.size) == ('hiking', 'hobby')
        preds = graph_batch.copy_preds(new_targets, vocab.size)
        tokens = textint_map.int_to_text(preds[0], 'target')
        expected = ['work', ('alice', 'person'), ('hiking', 'hobby')]
        assert_equal(tokens, expected)

    @pytest.mark.only
    def test_checklist(self, graph_batch, vocab, metadata):
        alice = metadata.entity_map.to_ind(('alice', 'person')) + vocab.size
        hiking = metadata.entity_map.to_ind(('hiking', 'hobby')) + vocab.size
        preds = np.array([[alice],
                          [hiking]])
        cl = graph_batch.get_zero_checklists(1)[:, 0, :]
        graph_batch.update_checklist(preds, cl, vocab)
        alice_node_id = graph_batch.graphs[0].nodes.to_ind(('alice', 'person'))
        hiking_node_id = graph_batch.graphs[1].nodes.to_ind(('hiking', 'hobby'))
        assert cl[0][alice_node_id] == 1 and sum(cl[0]) == 1
        assert cl[1][hiking_node_id] == 1 and sum(cl[0]) == 1

        targets = np.array([[0, alice, 0, 0],
                            [1, hiking, hiking, 0]])
        cl = graph_batch.get_checklists(targets, vocab)
        assert cl.shape[:2] == (2, 4)
        assert np.sum(cl[:, 0, :]) == 0  # Initial cl is zero
        assert np.sum(cl[:, 1, :]) == 0  # When generating at time=2
        for t in (2, 3):
            assert cl[0, t, alice_node_id] == 1
            assert cl[1, t, hiking_node_id] == 1
