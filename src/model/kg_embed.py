'''
Knowledge graph embedding models that provide context during generation.
'''

from itertools import chain
import tensorflow as tf

class Graph(object):
    '''
    An abstract class of graph embedding that takes a KB and returns vectorized context.
    '''
    def __init__(self, schema, embed_size, path_len=3):
        # Entities and relations
        self.entity_types = schema.get_entities()
        self.attribute_types = schema.get_attributes()
        self.relations = ['has_type', 'has_attr']
        # Vocabulary
        self.label_to_ind = {v: i for i, v in enumerate(chain(self.entity_types.keys(), self.entity_types.values(), self.relations))}
        self.label_size = len(self.label_to_ind)

        # An path is a tuple (e_1, r_1, e_2, r_2, ...)
        self.paths = []
        self.path_len = path_len

        self.build_model(embed_size)

    def load(self, kb):
        '''
        Populate the graph.
        '''
        def path_to_ind(path):
            return tuple(map(lambda x: self.label_to_ind[x], path))

        for item in kb.items:
            person = None
            for attr_name, value in item.iteritems():
                attr_type = self.attribute_types[attr_name]
                # Type node
                self.paths.append(path_to_ind((value, 'has_type', attr_type)))
                # Entity node
                if attr_type == 'person':
                    person = value
                else:
                    self.paths.append(path_to_ind((person, 'has_attr', value)))

    def build_model(self):
        raise NotImplementedError

class CBOWGraph(Graph):
    def build_model(self, embed_size):
        '''
        CBOW embedding model.
        '''
        with tf.variable_scope('graph'):
            embedding  = tf.get_variable('embedding', [self.label_size, embed_size])
            # Input is a list of paths: n x path_len
            self.input_data = tf.placeholder(tf.int32, shape=[None, self.path_len])
            # CBOW
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)  # n x path_len x embed_size
            self.context = tf.reduce_sum(inputs, 1)  # n x embed_size

