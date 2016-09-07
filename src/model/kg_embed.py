'''
Knowledge graph embedding models that provide context during generation.
'''

import sys
import numpy as np
from itertools import chain
from collections import defaultdict
import tensorflow as tf

def add_kg_arguments(parser):
    parser.add_argument('--kg-model', default='cbow', help='Model name {cbow}')
    parser.add_argument('--kg-embed-size', default=128, help='Knowledge graph embedding size')
    parser.add_argument('--entity-hist-len', type=int, default=10, help='Number of past words to search for entities')
    parser.add_argument('--entity-cache-size', type=int, default=2, help='Number of entities to remember (this is more of a performance concern; ideally we can remember all entities within the history)')
    parser.add_argument('--train-utterance', action='store_true', default=False, help='Whether to backpropogate error from utterance node')

# TODO: len_path = 4 (2-hop)

class Graph(object):
    '''
    An abstract class of graph embedding that takes a KB and returns vectorized context.
    '''
    def __init__(self, schema, lexicon, embed_size, utterance_size, entity_cache_size=2, entity_hist_len=10, path_len=3, train_utterance=False, scope=None):
        '''
        embed_size: embedding size for entities
        '''
        # Entities and relations
        self.total_num_entities = len(lexicon.entity_to_id)
        print 'Total number of entities:', self.total_num_entities
        self.attribute_types = schema.get_attributes()
        relations = self.attribute_types.keys()
        # Inverse relations
        relations.extend((self.inv_rel(r) for r in self.attribute_types))
        print 'Number of relations:', len(relations)

        # Vocabulary
        # Share entity mapping with lexicon
        self.label_to_ind = lexicon.entity_to_id
        # Add edge labels (relations)
        for i, v in enumerate(relations):
            self.label_to_ind[v.lower()] = self.total_num_entities + i
        self.ind_to_label = {v: k for k, v in self.label_to_ind.iteritems()}
        self.label_size = len(self.label_to_ind)

        # An path is a tuple (e_1, r_1, e_2, r_2, ...)
        self.path_len = path_len
        self.entity_cache_size = entity_cache_size
        self.entity_hist_len = entity_hist_len

        self.embed_size = embed_size
        # size of input utterances (RNN output size)
        self.utterance_size = utterance_size
        self.train_utterance = train_utterance
        if train_utterance:
            self.update_utterance = self.update_utterance_trainable
        self.build_model(scope)

    def inv_rel(self, relation):
        '''
        Inverse relation
        '''
        return '*' + relation

    def get_entity_list(self, tokens):
        '''
        Input tokens is a list of words/tuples where tuples represent entities.
        Output is a list of entity list at each position.
        The entity list contains self.entity_cache_size number of mentioned entities counting from the current position backwards.
        -1 means no entity.
        E.g. with cache_size = 2, I went to Stanford and MIT . =>
        [(-1, -1), (-1, -1), (-1, -1), (-1, Stanford), (Stanford, -1), (Stanford, MIT)]
        '''
        N = len(tokens)
        tokens = [''] * (self.entity_hist_len - 1) + tokens
        entity_list = []

        for i in xrange(N):
            entity_list.append(self.get_entities(tokens[i:i+self.entity_hist_len]))
        return entity_list

    def convert_entity(self, entities):
        '''
        Convert the entity list to indices to be updated in the utterance matrix.
        '''
        return [[[i, j] for j in range(self.utterance_size)] for i in entities]

    def get_entities(self, tokens):
        '''
        Return all entities in tokens. Pad -1 to fit in entity_cache_size.
        '''
        entities = [self.label_to_ind[x] for x in tokens if not isinstance(x, basestring)]
        # Remove duplicated entities
        entities = list(set(entities))
        if len(entities) < self.entity_cache_size:
            for i in xrange(self.entity_cache_size - len(entities)):
                entities.insert(0, -1)
        else:
            entities = entities[-1*self.entity_cache_size:]
        if self.train_utterance:
            entities = self.convert_entity(entities)
        return entities

    def get_paths(self, kb):
        '''
        Return 3-tuple paths in the KB: (entity, relation, entity),
        e.g., (Alice, Hobby, reading)
        '''
        def path_to_ind(path):
            return tuple(map(lambda x: self.label_to_ind[x], path))
        paths = []
        for item in kb.items:
            person = (item['Name'].lower(), 'person')
            for attr_name, value in item.iteritems():
                # Type node
                attr_type = self.attribute_types[attr_name]
                #self.paths.append(path_to_ind((value, 'has_type', attr_type)))
                # Entity node
                if attr_name != 'Name':
                    attr_name = attr_name.lower()
                    attr = (value.lower(), attr_type)
                    paths.append(path_to_ind((person, attr_name, attr)))
                    paths.append(path_to_ind((attr, self.inv_rel(attr_name), person)))
        return np.array(paths)

    def get_node_paths(self, paths):
        '''
        Return a list of nodes (all entities in the current kb)
        and its incident edges (as index in paths).
        '''
        node_paths = defaultdict(lambda : [False] * len(paths))
        for i, path in enumerate(paths):
            e1, r, e2 = path
            node_paths[e1][i] = True
        paths = []  # Paths of each node
        self.nodes = []  # Entity id of each node
        for node_id, path in node_paths.iteritems():
            self.nodes.append(node_id)
            paths.append(path)
        return paths

    def load(self, kb):
        '''
        Populate the graph.
        '''
        paths = np.array(self.get_paths(kb))
        node_paths = np.array(self.get_node_paths(paths))
        features = self.get_features(kb)
        # Size of the output attention scores (for copy)
        self.num_entities = len(self.nodes)
        entity_size = np.array(range(self.num_entities)).reshape(1, -1)
        # Output copied entities are in [0, num_nodes], need to map to global entity id
        self.local_to_global_entity = {i: entity_id for i, entity_id in enumerate(self.nodes)}
        self.global_to_local_entity = {v: k for k, v in self.local_to_global_entity.iteritems()}
        return (paths, node_paths, features, entity_size)

    def path_embedding(self, inputs):
        '''
        Input is a list of paths: n x path_len
        Return a tensor of shape n x embed_size
        '''
        raise NotImplementedError

    def build_model(self, scope):
        with tf.variable_scope(scope or type(self).__name__):
            # Input is a list of paths: n x path_len
            # a list of node paths: num_entities x path_len
            # features for each node
            # entity_size: tell RNN the output size (only useful for copy)
            self.input_data = (tf.placeholder(tf.int32, shape=[None, self.path_len]), tf.placeholder(tf.bool, shape=[None, None]), tf.placeholder(tf.float32, shape=[self.label_size, 1]), tf.placeholder(tf.float32, shape=[1, None]))
            paths, node_paths, self.features, self.entity_size = self.input_data
            paths = self.path_embedding(paths)  # n x embed_size
            # Node embedding is average of path embedding
            def average_paths(path_ind):
                # squeeze because the output of where is (?, 1)
                return tf.reduce_mean(tf.gather(paths, tf.squeeze(tf.where(path_ind), [1])), 0)
            nodes = tf.map_fn(average_paths, node_paths, dtype=tf.float32)  # num_entities x embed_size
            # NOTE: batch_size = 1!
            # context: batch_size x context_len x embed_size
            self.context = tf.expand_dims(nodes, 0)

    # TODO: add feature size as an argument
    def get_features(self, kb):
        features = np.zeros([self.label_size, 1], dtype=np.float32)
        sorted_attr = kb.sorted_attr()
        N = len(sorted_attr)
        # TODO: what if same attr_value appear >1, e.g., mit has rankings corresponding to both undergrad and master schools
        for i, attr in enumerate(sorted_attr):
            attr_name, attr_value = attr[0]
            ind = self.label_to_ind[(attr_value.lower(), self.attribute_types[attr_name])]
            # TODO: normalization seems important here, check tf batch normalization options
            features[ind][0] = (i + 1) / float(N)
        return features

    def entity_embedding(self):
        with tf.variable_scope('EntityEmbedding'):
            # Static embedding for each entity
            # TODO: should we share this with the input embedding?
            entity = tf.get_variable('static', [self.label_size, self.embed_size])
            # Dynamic embedding for utterance related to each entity
            if self.train_utterance:
                self.utterances = tf.zeros([self.label_size, self.utterance_size])
            else:
                self.utterances = tf.get_variable('utterance', [self.label_size, self.utterance_size], trainable=False, initializer=tf.zeros_initializer)
            # Other features

            # Concatenate entity and utterance embedding
            # TODO: other options, e.g., sum, project
            entity_embedding = tf.concat(1, [entity, self.utterances, self.features])
            #entity_embedding = tf.concat(1, [entity, self.features])
            #entity_embedding = self.features
            return entity_embedding

    # TODO: update nodes not in kg?
    def update_utterance_trainable(self, indices, utterance):
        '''
        indices: batch_size x cache_size x utterance_size x 2
        '''
        # NOTE: assume batch_size = 1
        indices = tf.squeeze(indices, [0])
        utterance = tf.squeeze(utterance)
        def cond_to_dense(ind):
            def to_dense():
                return tf.sparse_to_dense(ind, self.utterances.get_shape(), utterance)
            def no_op():
                return tf.zeros(self.utterances.get_shape())
            return tf.cond(tf.less(ind[0][0], 0), no_op, to_dense)
        utterance_list = tf.map_fn(cond_to_dense, indices, dtype=tf.float32)
        self.utterances = self.utterances + tf.foldl(lambda a, x: a + x, utterance_list)
        return self.utterances

    def update_utterance(self, indices, utterance):
        '''
        Update entries in matrix self.utterances.
        indices corresponds to first dimensions in self.utterances.
        '''
        def cond_update(ind):
            def no_op():
                return self.utterances
            def update():
                return tf.scatter_update(self.utterances, ind, utterance)
            return tf.cond(tf.less(ind, tf.constant(0)), no_op, update)
        # NOTE: assumes batch_size = 1
        indices = tf.reshape(indices, [-1])
        utterance = tf.reshape(utterance, [-1])
        return tf.map_fn(cond_update, indices, dtype=tf.float32)

class CBOWGraph(Graph):
    '''
    CBOW embedding model.
    '''
    def path_embedding(self, inputs):
        with tf.variable_scope('PathEmbedding'):
            embedding = self.entity_embedding()
            embed_input = tf.nn.embedding_lookup(embedding, inputs)  # n x path_len x embed_size
            # context_size: the context matrix sent to RNN
            self.context_size = embedding.get_shape()[1]
            return tf.reduce_sum(embed_input, 1)  # n x embed_size

# test
if __name__ == '__main__':
    import argparse
    from basic.dataset import add_dataset_arguments, read_dataset
    from basic.schema import Schema
    from basic.scenario_db import ScenarioDB, add_scenario_arguments
    from basic.lexicon import Lexicon
    from basic.util import read_json
    import random
    from model.preprocess import DataGenerator

    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    add_scenario_arguments(parser)
    add_dataset_arguments(parser)
    args = parser.parse_args()
    random.seed(args.random_seed)

    schema = Schema(args.schema_path)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))
    dataset = read_dataset(scenario_db, args)
    lexicon = Lexicon(schema)

    data_generator = DataGenerator(dataset.train_examples, dataset.test_examples, None, lexicon, None)

    gen = data_generator.generator_train('train')
    agent, kb, inputs, targets, iswrite = gen.next()

    embed_size = 10
    kg = CBOWGraph(schema, lexicon, embed_size)
    input_data = kg.load(kb)
    paths, node_paths = input_data
    print kb.dump()
    print kg.label_to_ind
    print 'Example paths:'
    def path_to_str(path):
        return map(lambda x: kg.ind_to_label[x], path)
    for path in paths[:10]:
        print path_to_str(path)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        print 'num_path x path_len:', paths.shape
        print 'num_entities x num_path:', node_paths.shape
        for i, node_id in enumerate(kg.nodes):
            print 'NODE:', kg.ind_to_label[node_id]
            print 'PATHS:'
            for j, has_path in enumerate(node_paths[i]):
                if has_path:
                    print path_to_str(paths[j])
            break
        feed_dict = {kg.input_data: input_data}
        [context] = sess.run([kg.context], feed_dict=feed_dict)
        print 'batch_size x num_entities x embed_size:', context.shape
