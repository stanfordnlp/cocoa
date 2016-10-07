import tensorflow as tf
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell import _linear as linear

def add_graph_embed_arguments(parser):
    parser.add_argument('--node-embed-size', default=50, help='Knowledge graph node/subgraph embedding size')
    parser.add_argument('--edge-embed-size', default=20, help='Knowledge graph edge label embedding size')
    parser.add_argument('--entity-embed-size', default=20, help='Knowledge graph entity embedding size')
    parser.add_argument('--entity-cache-size', type=int, default=2, help='Number of entities to remember (this is more of a performance concern; ideally we can remember all entities within the history)')
    parser.add_argument('--use-entity-embedding', action='store_true', default=False, help='Whether to use entity embedding when compute node embeddings')
    parser.add_argument('--train-utterance', action='store_true', default=False, help='Whether to backpropogate error from utterance node')
    parser.add_argument('--mp-iter', type=int, default=2, help='Number of iterations of message passing on the graph')
    parser.add_argument('--combine-message', default='concat', help='How to combine propogated message {concat, sum}')

class GraphEmbed(object):
    '''
    Graph embedding model.
    '''
    def __init__(self, max_num_nodes, num_edge_labels, node_embed_size, edge_embed_size, utterance_size, feat_size, entity_cache_size=2, num_entities=None, entity_embed_size=None, use_entity_embedding=False, mp_iter=2, message='concat', scope=None):
        # The maximum number of entities that may appear in a single dialogue. This only
        # affects the size of the utterance matrix (because we cannot change its size dynamically).
        # Adding 1 because we want a dummy utterance node which is updated by default when
        # no other node needs to be udpated. This is to avoid using tf.cond().
        self.max_num_nodes = max_num_nodes + 1
        self.num_edge_labels = num_edge_labels
        self.node_embed_size = node_embed_size
        self.edge_embed_size = edge_embed_size
        self.utterance_size = utterance_size  # RNN output size
        self.feat_size = feat_size
        self.mp_iter = mp_iter  # number of message passing iterations
        self.message = message
        if use_entity_embedding:
            assert num_entities is not None
            self.num_entities = num_entities
            self.entity_embed_size = entity_embed_size
        self.use_entity_embedding = use_entity_embedding

        self.scope = scope
        self.build_model(scope)

        self.input_entity_shape = [1, None, self.entity_cache_size, self.utterance_size, 2]

    def _build_utterance_embedding(self):
        with tf.variable_scope('UtteranceEmbedding'):
            utterances = tf.zeros([self.max_num_nodes, self.utterance_size])
        return utterances

    def _expand_dim_entity(self, entities, dim_size):
        '''
        Expand a list of entity ids to indices to be updated in the utterance matrix.
        TODO: It's cleaner to do this tranformation in TF but I haven't found a good way.
        '''
        return [[[i, j] for j in xrange(dim_size)] for i in entities]

    def _normalize_entity(self, entities, max_len, pad):
        '''
        Put the entity lists to the same size by padding or truncating.
        '''
        if len(entities) < max_len:
            # Pad dummy entity
            for i in xrange(max_len - len(entities)):
                entities.insert(0, pad)
        else:
            # Take the most recent ones
            entities = entities[-1*max_len:]
        return entities

    def reshape_input_entity(self, entity_list):
        '''
        Preprocess so that it's ready to be used by update_utterance.
        '''
        pad = self.max_num_nodes
        max_len = self.entity_cache_size
        new_dim_size = self.utterance_size
        entity_list = [self._expand_dim_entity(\
                self._normalize_entity_list(e, max_len, pad), new_dim_size)\
                for e in entity_list]
        entity_list = np.asarray(entity_list, dtype=np.int32).reshape(1, -1, max_len, self.utterance_size, 2)
        return entity_list

    def build_model(self, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope('EdgeEmbedding'):
                self.edges = tf.get_variable('edge', [self.num_edge_labels, self.edge_embed_size])

            if self.use_entity_embedding:
                with tf.variable_scope('EntityEmbedding'):
                    self.entities = tf.get_variable('entity', [self.num_entities, self.entity_embed_size])

            self.utterances = self._build_utterance_embedding()

            # Inputs that specify the graph structure:
            # node_ids: an array of node_ids that links a node to its utterance embedding in self.utterance
            # paths: (node_id, edge_labe, node_id)
            # feats: node features. NOTE: feats[i] must corresponds to node_ids[i].
            self.input_data = (tf.placeholder(tf.int32, shape=[None]), \
                               tf.placeholder(tf.int32, shape=[None]), \
                               tf.placeholder(tf.int32, shape=[None, 3]), \
                               tf.placeholder(tf.float32, shape=[None, self.feat_size]))
            self.node_ids = self.input_data[0]

    def get_context(self, utterances=None):
        '''
        This part is separated from build_model because we will essentially re-build and
        re-comopute the context node at each time step due to potential change of utterances.
        '''
        if utterances is None:
            utterances = self.utterances
        node_ids, entity_ids, paths, node_feats = self.input_data
        with tf.variable_scope(self.scope or type(self).__name__):
            with tf.variable_scope('NodeEmbedding'):
                with tf.variable_scope('InitNodeEmbedding'):
                    if self.use_entity_embedding:
                        initial_node_embed = linear([tf.nn.embedding_lookup(self.entities, entity_ids), tf.nn.embedding_lookup(utterances, node_ids), node_feats], self.node_embed_size, True)
                    else:
                        initial_node_embed = linear([tf.nn.embedding_lookup(utterances, node_ids), node_feats], self.node_embed_size, True)
                # Message passing
                #node_embeds = tf.scan(lambda curr_embed, _: self.pass_message(curr_embed, paths), \
                #        tf.range(0, self.mp_iter), \
                #        initial_node_embed)
                node_embeds = self.pass_message(initial_node_embed, paths)

            if self.message == 'concat':
                #context = tf.concat(1, [initial_node_embed] + tf.unpack(node_embeds, axis=0))
                context = tf.concat(1, [initial_node_embed, node_embeds])
            elif self.message == 'sum':
                context = tf.add_n([initial_node_embed] + tf.unpack(node_embeds, axis=0))
            else:
                raise ValueError('Unknown message combining method')
        # NOTE: batch_size = 1!
        # context: batch_size x context_len x embed_size
        context = tf.expand_dims(context, 0)
        return context

    def pass_message(self, curr_embed, paths):
        '''
        Compute new node embeddings: sum of neighboring node embeddings.
        paths: (node_id, edge_label, node_id)  num_paths x 3
        '''
        edge_embeds = tf.nn.embedding_lookup(self.edges, paths[:, 1])
        node_embeds = tf.nn.embedding_lookup(curr_embed, paths[:, 2])
        path_embeds = tanh(linear([edge_embeds, node_embeds], self.node_embed_size, True))  # num_paths x embed_size

        def sum_paths(node_id):
            # Neighbroing paths of a node
            node_path = tf.squeeze(tf.where(tf.equal(paths[:, 0], node_id)), [1])
            # Whether this is a zero-degree node; if so, return a zero tensor
            pred = tf.equal(tf.shape(node_path)[0], 0)
            return tf.cond(pred, \
                    lambda : tf.zeros([self.node_embed_size]), \
                    lambda : tf.reduce_sum(tf.gather(path_embeds, node_path), 0))
        new_embeds = tf.map_fn(sum_paths, self.node_ids, dtype=tf.float32)
        return new_embeds

    def update_utterance(self, indices, utterance):
        '''
        indices: batch_size x cache_size x utterance_size x 2
        specifies each index to update in tensor self.utterances
        '''
        indices = tf.squeeze(indices, [0])  # NOTE: assume batch_size = 1
        utterance = tf.squeeze(utterance)
        # Duplicate utterance for each entity
        num_entities = self.entity_cache_size
        utterances = tf.reshape(tf.tile(utterance, num_entities), [num_entities, self.utterance_size])
        # TODO: test vector sparse indices (first dimension of dense)
        delta = tf.sparse_to_dense(
        def cond_to_dense(ind):
            def to_dense():
                return tf.sparse_to_dense(ind, self.utterances.get_shape(), utterance)
            def no_op():
                return tf.zeros(self.utterances.get_shape())
            return tf.cond(tf.less(ind[0][0], 0), no_op, to_dense)
        utterance_list = tf.map_fn(cond_to_dense, indices, dtype=tf.float32)
        self.utterances = self.utterances + tf.foldl(lambda a, x: a + x, utterance_list)
        return self.utterances

    def test_mp(self, sess, feed_dict):
        with tf.variable_scope('test_mp'):
            embed0 = tf.constant(np.ones([3, 4]), dtype=tf.float32)
            print 'set edge embedding to zero'
            self.edges = tf.constant(np.zeros([1, 4]), dtype=tf.float32)
            embed1 = self.pass_message(embed0, self.input_data[2])
            tf.get_variable_scope().reuse_variables()
            embed2 = self.pass_message(embed1, self.input_data[2])

        tf.initialize_all_variables().run()
        [result0, result1, result2] = sess.run([embed0, embed1, embed2], feed_dict=feed_dict)
        print 'initial embed:\n', result0
        print 'propogate from node 1 and 2 to node 0'
        print 'message at t=1:\n', result1
        print 'message at t=2:\n', result2

    def test(self, sess, feed_dict):
        with tf.variable_scope('test'):
            context = self.get_context()
            tf.initialize_all_variables().run()
            [result] = sess.run([context], feed_dict=feed_dict)
            print 'node embeddings:\n', result

            print 'update utterances'
            tf.get_variable_scope().reuse_variables()
            utterance = tf.constant(np.ones([1, 4]), dtype=tf.float32)
            indices = tf.constant(np.array([[1, 0], [1, 1], [1, 2], [1, 3]]).reshape([1, 1, 4, 2]), dtype=tf.int32)
            utterances = self.update_utterance(indices, utterance)
            context = self.get_context(utterances)
            [result] = sess.run([context], feed_dict=feed_dict)
            print 'node embeddings:\n', result

class GraphEmbedStaticUtterance(GraphEmbed):
    '''
    Graph embedding model.
    The utterance is copied from the RNN state but no gradient is backpropogated back to the RNN.
    '''
    def __init__(*args, **kwargs):
        super(GraphEmbedStaticUtterance, self).__init__(*args, **kwargs)
        self.input_entity_shape = [1, None, self.entity_cache_size]

    def _build_utterance_embedding(self):
        with tf.variable_scope('UtteranceEmbedding'):
            utterances = tf.get_variable('utterance', [self.max_num_nodes, self.utterance_size], trainable=False, initializer=tf.zeros_initializer)
        return utterances

    def reshape_input_entity(self, entity_list):
        '''
        Preprocess so that it's ready to be used by update_utterance.
        '''
        pad = self.max_num_nodes
        max_len = self.entity_cache_size
        entity_list = [self._normalize_entity_list(e, max_len, pad) for e in entity_list]
        entity_list = np.asarray(entity_list, dtype=np.int32).reshape(1, -1, 2)
        return entity_list

    def update_utterance(self, indices, utterance):
        '''
        Update entries in matrix self.utterances.
        indices corresponds to first dimensions in self.utterances.
        '''
        def cond_update(ind):
            def no_op():
                return self.utterances
            def update():
                # in-place update
                return tf.scatter_update(self.utterances, ind, utterance)
            return tf.cond(tf.less(ind, tf.constant(0)), no_op, update)
        # NOTE: assumes batch_size = 1
        indices = tf.reshape(indices, [-1])
        utterance = tf.reshape(utterance, [-1])
        tf.map_fn(cond_update, indices, dtype=tf.float32)
        return self.utterances


if __name__ == '__main__':
    import numpy as np
    max_num_nodes = 10
    num_edge_labels = 1
    node_embed_size = 4
    edge_embed_size = 4
    utterance_size = 4
    feat_size = 1

    tf.reset_default_graph()
    tf.set_random_seed(1)
    graph_embed = GraphEmbed(max_num_nodes, num_edge_labels, node_embed_size, edge_embed_size, utterance_size, feat_size, mp_iter=2, message='concat', scope=None)

    num_nodes = 3
    node_ids = np.array([0, 1, 2])
    paths = np.array([[0, 0, 1], [0, 0, 2]])
    node_feats = np.zeros([num_nodes, feat_size])
    feed_dict = {graph_embed.input_data: (node_ids, entity_ids, paths, node_feats)}

    with tf.Session() as sess:
        graph_embed.test(sess, feed_dict)
        graph_embed.test_mp(sess, feed_dict)

