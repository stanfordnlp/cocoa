'''
Knowledge graph embedding models that provide context during generation.
'''

from itertools import chain
import tensorflow as tf

def add_kg_arguments(parser):
    parser.add_argument('--kg-model', default='cbow', help='Model name {cbow}')
    parser.add_argument('--kg-embed-size', default=128, help='Knowledge graph embedding size')

class Graph(object):
    '''
    An abstract class of graph embedding that takes a KB and returns vectorized context.
    '''
    def __init__(self, schema, embed_size, path_len=3, scope=None):
        # Entities and relations
        self.entity_types = schema.get_entities()
        self.attribute_types = schema.get_attributes()
        self.relations = ['has_type', 'has_attr']
        # Vocabulary
        # NOTE: entities are all lowercased
        self.label_to_ind = {v.lower(): i for i, v in enumerate(chain(self.entity_types.keys(), set(self.entity_types.values()), self.relations))}
        self.ind_to_label = {v: k for k, v in self.label_to_ind.iteritems()}
        self.label_size = len(self.label_to_ind)

        # An path is a tuple (e_1, r_1, e_2, r_2, ...)
        self.paths = []
        self.path_len = path_len

        self.embed_size = embed_size
        self.build_model(scope)

    def load(self, kb):
        '''
        Populate the graph.
        '''
        def path_to_ind(path):
            return tuple(map(lambda x: self.label_to_ind[x.lower()], path))

        for item in kb.items:
            person = None
            person_attr_values = []
            for attr_name, value in item.iteritems():
                attr_type = self.attribute_types[attr_name]
                # Type node
                self.paths.append(path_to_ind((value, 'has_type', attr_type)))
                # Entity node
                if attr_type == 'person':
                    person = value
                else:
                    person_attr_values.append(value)
            assert person is not None
            for value in person_attr_values:
                self.paths.append(path_to_ind((person, 'has_attr', value)))

    def build_model(self):
        raise NotImplementedError

class CBOWGraph(Graph):
    def build_model(self, scope=None):
        '''
        CBOW embedding model.
        '''
        with tf.variable_scope(scope or type(self).__name__):
            #self.embedding  = tf.get_variable('embedding', [self.label_size, self.embed_size])
            # Input is a list of paths: n x path_len
            self.input_data = tf.placeholder(tf.int32, shape=[None, self.path_len])
            ## CBOW
            #inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)  # n x path_len x embed_size
            #self.context = tf.reduce_sum(inputs, 1)  # n x embed_size
            ## NOTE: batch_size = 1!
            ## context: batch_size x context_len x embed_size
            #self.context = tf.expand_dims(self.context, 0)

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
    kg = CBOWGraph(schema, embed_size)
    kg.load(kb)
    print kb.dump()
    print kg.label_to_ind
    print 'PATHS:\n'
    for path in kg.paths[:10]:
        print map(lambda x: kg.ind_to_label[x], path)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        [context] = sess.run([kg.context], feed_dict={kg.input_data: kg.paths})
        print context.shape
