from collections import defaultdict
import numpy as np
from itertools import izip, islice
from vocab import is_entity, Vocabulary

def add_graph_arguments(parser):
    parser.add_argument('--num-items', type=int, default=10, help='Number of items in each KB')
    parser.add_argument('--entity-hist-len', type=int, default=10, help='Number of past words to search for entities')

class Graph(object):
    '''
    Maintain a (dynamic) knowledge graph of the agent.
    '''
    setup = False
    @classmethod
    def static_init(cls, schema, entity_map, relation_map, max_degree=10, entity_hist_len=10):
        cls.attribute_types = schema.get_attributes()
        cls.entity_map = entity_map
        # Add new relations (edge labels)
        relation_map.add_word('has')
        relation_map.add_words([cls.inv_rel(r) for r in relation_map.word_to_ind])
        cls.relation_map = relation_map
        cls.entity_hist_len = entity_hist_len

        # Node features {feat_name: (offset, feat_size)}
        # degree: 0-10
        # node_type: entity, item, attr
        degree_size = max_degree + 1
        cls.feat_inds = {'degree': (0, degree_size), 'node_type': (degree_size, 3)}
        cls.feat_size = sum([v[1] for v in cls.feat_inds.values()])

        cls.setup = True

    @classmethod
    def inv_rel(cls, relation):
        return '*' + relation

    @classmethod
    def get_feat_vec(cls, raw_feats):
        '''
        Input: a list of features [degree, node_type] for each node
        Output: one-hot encoded numpy feature matrix
        '''
        f = np.zeros([len(raw_feats), cls.feat_size])

        def get_index(feat_name, feat_value):
            offset, size = cls.feat_inds[feat_name]
            assert feat_value < size
            return offset + feat_value

        for i, (degree, node_type) in enumerate(raw_feats):
            f[i][get_index('degree', degree)] = 1

            if node_type == 'item':
                node_type = 0
            elif node_type == 'attr':
                node_type = 1
            else:
                node_type = 2
            f[i][get_index('node_type', node_type)] = 1

        return f

    def __init__(self, kb):
        if not Graph.setup:
            raise Exception('Error: Instantiating Graph before initializing its static variables.')
        self.nodes = Vocabulary(unk=False)
        self.paths = []
        self.load_kb(kb)

        # Input data to feed_dict
        self.node_ids = np.arange(self.nodes.size, dtype=np.int32)
        self.entity_ids = np.array([self.entity_map.to_ind(self.nodes.to_word(i)) for i in xrange(self.nodes.size)], dtype=np.int32)
        self.paths = np.array(self.paths, dtype=np.int32)
        self.feats = self.get_features()

        # Entity/token sequence in the dialogue
        self.entities = []

    def get_input_data(self):
        '''
        Return feed_dict data to the GraphEmbed model.
        '''
        assert self.node_ids.shape[0] == self.feats.shape[0]
        return (self.node_ids, self.entity_ids, self.paths, self.feats)

    def _add_edge(self, node1, relation, node2):
        node1_id = self.nodes.to_ind(node1)
        node2_id = self.nodes.to_ind(node2)
        rel = self.relation_map.to_ind(relation)
        irel = self.relation_map.to_ind(self.inv_rel(relation))
        self.paths.append((node1_id, rel, node2_id))
        self.paths.append((node2_id, irel, node1_id))

    def load_kb(self, kb):
        attr_ents = defaultdict(set)  # Entities of each attribute
        for i, item in enumerate(kb.items):
            # Item nodes
            item_node = ('item-%d' % i, 'item')
            self.nodes.add_word(item_node)
            for attr_name, value in item.iteritems():
                type_ = self.attribute_types[attr_name]
                attr_name = attr_name.lower()
                value = value.lower()
                # Attribute nodes
                attr_node = (attr_name, 'attr')
                self.nodes.add_word(attr_node)
                # Entity nodes
                entity_node = (value, type_)
                self.nodes.add_word(entity_node)
                # Path: item has_attr entity
                self._add_edge(item_node, attr_name, entity_node)
                attr_ents[attr_node].add(entity_node)
        # Path: attr has entity
        for attr_node, ent_set in attr_ents.iteritems():
            for entity_node in ent_set:
                self._add_edge(attr_node, 'has', entity_node)
        self.paths = np.array(self.paths, dtype=np.int32)

    def read_utterance(self, tokens):
        '''
        Map entities to node ids and tokens to -1. Add new nodes if needed.
        '''
        new_entities = set([x for x in tokens if is_entity(x) and not self.nodes.has(x)])
        self.add_entity_nodes(new_entities)
        node_ids = (self.nodes.to_ind(x) if is_entity(x) else -1 for x in tokens)
        self.entities.extend(node_ids)

    def add_entity_nodes(self, entities):
        # Add nodes
        self.nodes.add_words(entities)
        # TODO: Maybe use concatenate too..
        self.node_ids = np.arange(self.nodes.size, dtype=np.int32)
        self.entity_ids = np.array([self.entity_map.to_ind(self.nodes.to_word(i)) for i in xrange(self.nodes.size)], dtype=np.int32)
        # Update feats: degree=0, node_type=entity
        feats = [[0, 'entity'] for _ in entities]
        new_feat_vec = self.get_feat_vec(feats)
        self.feats = np.concatenate((self.feats, new_feat_vec), axis=0)
        # Paths do not change

    def get_entity_list(self, last_n=None):
        '''
        Input: return entity_list for the n most recent tokens received
        Output: a list of entity list at each position of received entities
        - E.g. I went to Stanford and MIT . => [[], [], [], [Stanford], [Stanford], [Stanford, MIT]]
        '''
        N = len(self.entities)
        if not last_n:
            position = xrange(N)
        else:
            assert last_n <= N
            position = (N - i for i in xrange(last_n, 0, -1))
        entity_list = [self.get_entities(max(0, i-self.entity_hist_len), i+1) for i in position]
        return entity_list

    def get_entities(self, start, end):
        '''
        Return all entity ids (from self.nodes) in [start, end).
        '''
        # Filter tokens and remove duplicated entities
        seen = set()
        entities = [entity for entity in islice(self.entities, start, end) if \
                entity != -1 and not (entity in seen or seen.add(entity))]
        return entities

    def get_features(self):
        feats = [[0, self.nodes.to_word(i)[1]] for i in xrange(self.nodes.size)]
        # Compute degree of each node
        for path in self.paths:
            n1, r, n2 = path
            feats[n1][0] += 1
        return self.get_feat_vec(feats)

    def copy_target(self, targets, iswrite, vocab):
        # Don't change the original targets, will be used later
        new_targets = np.copy(targets)
        for target, write in izip(new_targets, iswrite):
            for i, (t, w) in enumerate(izip(target, write)):
                if w:
                    # TODO: what if this is an UNK. inputs and targets from data generator
                    # should be tokens and entities. and they should be mapped in update_feed_dict.
                    token = vocab.to_word(t)
                    if is_entity(token):
                        try:
                            target[i] = vocab.size + self.nodes.to_ind(token)
                        except KeyError:
                            # TODO: for real data, this may not be a bug: we might have failed to
                            # detect a previously mentioned entity.
                            import sys
                            print token
                            self.nodes.dump()
                            sys.exit()
        return new_targets

    def copy_output(self, outputs, vocab):
        for output in outputs:
            for i, pred in enumerate(output):
                if pred >= vocab.size:
                    entity = self.nodes.to_word(pred - vocab.size)
                    # TODO: again, this could be an UNK. output shoule be tokens/entities instead of numbers
                    output[i] = vocab.to_ind(entity)
        return outputs

if __name__ == '__main__':
    from basic.schema import Schema
    from model.preprocess import build_schema_mappings
    from basic.kb import KB

    schema = Schema('data/friends-schema.json')
    entity_map, relation_map = build_schema_mappings(schema)
    max_degree = 3

    items = [{'Name': 'Alice', 'Company': 'Microsoft', 'Hobby': 'hiking'},\
             {'Name': 'Bob', 'Company': 'Apple', 'Hobby': 'hiking'}]
    kb = KB.from_dict(schema, items)

    Graph.static_init(schema, entity_map, relation_map, max_degree)
    graph = Graph(kb)

    # Basic tests
    print '%d nodes in graph:' % graph.nodes.size
    graph.nodes.dump()

    print '%d paths in graph:' % len(graph.paths)
    for path in graph.paths:
        n1, r, n2 = path
        print graph.nodes.to_word(n1), graph.relation_map.to_word(r), graph.nodes.to_word(n2)

    print 'Node features:'
    features = graph.get_features()
    print features.shape
    print features

    # Test changing graph structure
    print 'Add new entities:'
    graph.add_entity_nodes([('facebook', 'company')])
    print 'New nodes and features:'
    graph.nodes.dump()
    assert np.array_equal(graph.feats, graph.get_features())
    assert graph.feats.shape[0] == graph.nodes.size

    graph.read_utterance(['alice', 'works', 'at', ('google', 'company')])
    print 'Received entities:'
    print graph.entities
    graph.nodes.dump()
    assert graph.nodes.size == graph.feats.shape[0]
    print len(graph.entities)
    print graph.get_entity_list(1)

    # Test copy
    vocab = Vocabulary()
    words = [('alice', 'person'), ('hiking', 'hobby')]
    vocab.add_words(words)

    print 'Copy targets:'
    targets = np.array([map(vocab.to_ind, words)])
    iswrite = np.array([[False, True]])
    new_targets = graph.copy_target(targets, iswrite, vocab)
    assert new_targets[0][0] == targets[0][0]
    assert new_targets[0][1] == vocab.size + graph.nodes.to_ind(('hiking', 'hobby'))

    print 'Copy outputs:'
    assert np.array_equal(graph.copy_output(new_targets, vocab), targets)
