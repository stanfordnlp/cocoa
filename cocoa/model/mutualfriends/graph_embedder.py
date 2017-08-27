import tensorflow as tf
from tensorflow.python.ops.math_ops import tanh
from cocoa.model.util import _linear as linear
from cocoa.model.util import batch_embedding_lookup, batch_linear, EPS

def add_graph_embed_arguments(parser):
    parser.add_argument('--node-embed-size', type=int, default=10, help='Knowledge graph node/subgraph embedding size')
    parser.add_argument('--edge-embed-size', type=int, default=10, help='Knowledge graph edge label embedding size')
    parser.add_argument('--entity-embed-size', type=int, default=10, help='Knowledge graph entity embedding size')
    parser.add_argument('--entity-cache-size', type=int, default=2, help='Number of entities to remember (this is more of a performance concern; ideally we can remember all entities within the history)')
    parser.add_argument('--use-entity-embedding', action='store_true', default=False, help='Whether to use entity embedding when compute node embeddings')
    parser.add_argument('--mp-iters', type=int, default=2, help='Number of iterations of message passing on the graph')
    parser.add_argument('--utterance-decay', type=float, default=1, help='Decay of old utterance embedding over time')
    parser.add_argument('--learned-utterance-decay', default=False, action='store_true', help='Learning weight to combine old and new utterances')
    parser.add_argument('--msg-aggregation', default='sum', choices=['sum', 'max', 'avg'], help='How to aggregate messages from neighbors')

activation = tf.tanh

class GraphEmbedder(object):
    '''
    Graph embedding model.
    '''
    def __init__(self, config, scope=None):
        self.config = config
        self.scope = scope
        self.context_initialized = False
        self.update_initialized = False
        self.build_model(scope)

    def build_model(self, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope('EdgeEmbedding'):
                self.edge_embedding = tf.get_variable('edge', [self.config.num_edge_labels, self.config.edge_embed_size])

            if self.config.use_entity_embedding:
                with tf.variable_scope('EntityEmbedding'):
                    self.entity_embedding = tf.get_variable('entity', [self.config.num_entities, self.config.entity_embed_size])

            with tf.name_scope('Inputs'):
                # Nodes in the Graph, id is row index in utterances.
                # The number of nodes can vary in each batch.
                node_ids = tf.placeholder(tf.int32, shape=[None, None], name='node_ids')
                mask = tf.placeholder(tf.bool, shape=[None, None], name='mask')

                # Entity ids used for look up in entity_embedding when use_entity_embedding.
                # NOTE: node_ids is local; it's essentially range(number of nodes). entity_ids
                # use the global entity mapping.
                entity_ids = tf.placeholder(tf.int32, shape=[None, None], name='entity_ids')

                # A path is a tuple of (node_id, edge_label, node_id)
                # NOTE: we assume the first path is always a padding path (NODE_PAD, EDGE_PAD,
                # NODE_PAD) when computing mask in pass_message
                # The number of paths can vary in each batch.
                paths = tf.placeholder(tf.int32, shape=[None, None, 3], name='paths')

                # Each node has a list of paths starting from that node. path id is row index
                # in paths. Paths of padded nodes are PATH_PAD.
                node_paths = tf.placeholder(tf.int32, shape=[None, None, None], name='node_paths')

                # Node features. NOTE: feats[i] must corresponds to node_ids[i]
                node_feats = tf.placeholder(tf.float32, shape=[None, None, self.config.feat_size], name='node_feats')

                self.input_data = (node_ids, mask, entity_ids, paths, node_paths, node_feats)
                # TODO:
                self.node_ids, self.mask, self.entity_ids, self.paths, self.node_paths, self.node_feats = self.input_data

            # This will be used by GraphDecoder to figure out the shape of the output attention scores
            self.node_ids = self.input_data[0]

    def get_feed_dict(self, **kwargs):
        feed_dict = kwargs.pop('feed_dict', {})
        feed_dict[self.node_ids] = kwargs.pop('node_ids')
        feed_dict[self.mask] = kwargs.pop('mask')
        feed_dict[self.entity_ids] = kwargs.pop('entity_ids')
        feed_dict[self.paths] = kwargs.pop('paths')
        feed_dict[self.node_paths] = kwargs.pop('node_paths')
        feed_dict[self.node_feats] = kwargs.pop('node_feats')
        return feed_dict

    def get_context(self, utterances):
        '''
        Compute embedding of each node as context for the attention model.
        utterances: current utterance embeddings from the dialogue history
        '''
        node_ids, mask, entity_ids, paths, node_paths, node_feats = self.input_data
        with tf.variable_scope(self.scope or type(self).__name__):
            with tf.variable_scope('NodeEmbedding'):
                with tf.variable_scope('InitNodeEmbedding') as scope:
                    # It saves some reshapes to do batch_linear and batch_embedding_lookup
                    # together, but this way is clearer.
                    if self.config.use_entity_embedding:
                        initial_node_embed = tf.concat(2,
                                [tf.nn.embedding_lookup(self.entity_embedding, entity_ids),
                                 batch_embedding_lookup(utterances[0], node_ids),
                                 batch_embedding_lookup(utterances[1], node_ids),
                                 node_feats])
                    else:
                        initial_node_embed = tf.concat(2,
                                [batch_embedding_lookup(utterances[0], node_ids),
                                 batch_embedding_lookup(utterances[1], node_ids),
                                 node_feats])
                    scope.reuse_variables()

                # Message passing
                def mp(curr_node_embedding):
                    messages = self.embed_path(curr_node_embedding, self.edge_embedding, paths)
                    return self.pass_message(messages, node_paths, self.config.pad_path_id)

                node_embeds = [initial_node_embed]
                if self.config.mp_iters > 0:
                    # NOTE: initial MP uses different parameters because the node_embed_size is different
                    with tf.variable_scope('InitialMP'):
                        node_embeds.append(mp(node_embeds[-1]))
                    for i in xrange(self.config.mp_iters-1):
                        if i > 0:
                            tf.get_variable_scope().reuse_variables()
                        node_embeds.append(mp(node_embeds[-1]))

        context = tf.concat(2, node_embeds)

        self.context_initialized = True
        return context, mask

    def embed_path(self, node_embedding, edge_embedding, paths):
        '''
        Compute embedding of a path (edge_label, node_id).
        node_embedding: (batch_size, num_nodes, node_embed_size)
        edge_embedding: (num_edge_label, edge_embed_size)
        paths: each path is a tuple of (node_id, edge_label, node_id).
        (batch_size, num_paths, 3)
        '''
        edge_embeds = tf.nn.embedding_lookup(edge_embedding, paths[:, :, 1])
        node_embeds = batch_embedding_lookup(node_embedding, paths[:, :, 2])
        path_embed_size = self.config.node_embed_size
        path_embeds = activation(batch_linear([edge_embeds, node_embeds], path_embed_size, True))
        return path_embeds

    def pass_message(self, path_embeds, neighbors, padded_path=0):
        '''
        Compute new node embeddings by summing path embeddings (message) of neighboring nodes.
        neighbors: ids of neighboring paths of each node where id is row index in path_embeds
        (batch_size, num_nodes, num_neighbors)
        path_embeds: (batch_size, num_paths, path_embed_size)
        PATH_PAD: if a node is not incident to any edge, its path ids in neighbors are PATH_PAD
        '''
        # Mask padded nodes in neighbors
        # NOTE: although we mask padded nodes in get_context, we still need to mask neighbors
        # for entities not in the KB but mentioned by the partner. These are dangling nodes
        # and should not have messages passed in.
        mask = tf.to_float(tf.not_equal(neighbors, tf.constant(padded_path)))  # (batch_size, num_nodes, num_neighbors)
        num_neighbors = tf.reduce_sum(tf.cast(mask, tf.float32), 2, keep_dims=True) + EPS

        # Use static shape when possible
        shape = tf.shape(neighbors)
        batch_size, num_nodes, _ = neighbors.get_shape().as_list()
        batch_size = batch_size or shape[0]
        num_nodes = num_nodes or shape[1]
        path_embed_size = path_embeds.get_shape().as_list()[-1]

        # Gather neighboring path embeddings
        neighbors = tf.reshape(neighbors, [batch_size, -1])  # (batch_size, num_nodes x num_neighbors)
        embeds = batch_embedding_lookup(path_embeds, neighbors)  # (batch_size, num_nodes x num_neighbors, path_embed_size)
        embeds = tf.reshape(embeds, [batch_size, num_nodes, -1, path_embed_size])
        mask = tf.expand_dims(mask, 3)  # (batch_size, num_nodes, num_neighbors, 1)
        embeds = embeds * mask

        # (batch_size, num_nodes, path_embed_size)
        if self.config.msg_agg == 'sum':
            new_node_embeds = tf.reduce_sum(embeds, 2)
        elif self.config.msg_agg == 'avg':
            new_node_embeds = tf.reduce_sum(embeds, 2) / num_neighbors
        elif self.config.msg_agg == 'max':
            new_node_embeds = tf.reduce_max(embeds, 2)
        else:
            raise ValueError('Unknown message aggregation method')

        return new_node_embeds

    def update_utterance(self, entity_indices, utterance, curr_utterances, utterance_id):
        new_utterances = []
        for i, u in enumerate(curr_utterances):
            if i == utterance_id:
                new_utterances.append(self._update_utterance(entity_indices, utterance, u))
            else:
                new_utterances.append(u)
        return tuple(new_utterances)

    def _update_utterance(self, entity_indices, utterance, curr_utterances):
        '''
        We first transform utterance into a dense matrix of the same size as curr_utterances,
        then return their sum.
        entity_indices: entity ids correponding to rows to be updated in the curr_utterances
        (batch_size, entity_cache_size)
        utterance: hidden states from the RNN
        (batch_size, utterance_size)
        NOTE: each curr_utterance matrix should have a row (e.g. the last one) as padded utterance.
        Padded entities in entity_indices corresponds to the padded utterance. This is handled
        by GraphBatch during construnction of the input data.
        '''
        entity_inds_shape = tf.shape(entity_indices)
        B = entity_inds_shape[0]  # batch_size is a variable
        E = entity_inds_shape[1]  # number of entities to be updated
        U = self.config.utterance_size
        # Construct indices corresponding to each entry to be updated in self.utterances
        # self.utterance has shape (batch_size, num_nodes, utterance_size)
        # Therefore each row in the indices matrix specifies (batch_id, node_id, utterance_dim)
        batch_inds = tf.reshape(tf.tile(tf.reshape(tf.range(B), [-1, 1]), [1, E*U]), [-1, 1])
        node_inds = tf.reshape(tf.tile(tf.reshape(entity_indices, [-1, 1]), [1, U]), [-1, 1])
        utterance_inds = tf.reshape(tf.tile(tf.range(U), [E*B]), [-1, 1])
        inds = tf.concat(1, [batch_inds, node_inds, utterance_inds])

        # Repeat utterance for each entity
        utterance = tf.reshape(tf.tile(utterance, [1, E]), [-1])
        new_utterance = tf.sparse_to_dense(inds, tf.shape(curr_utterances), utterance, validate_indices=False)

        if self.config.learned_decay:
            with tf.variable_scope('UpdateUtterance', reuse=self.update_initialized):
                weight = tf.sigmoid(batch_linear(tf.concat(2, [curr_utterances, new_utterance]), 1, True))  # (batch_size, num_nodes, 1)
                if not self.update_initialized:
                    self.update_initialized = True


        if self.config.learned_decay:
            return tf.mul(1 - weight, curr_utterances) + tf.mul(weight, new_utterance)
        else:
            return curr_utterances * self.config.decay + new_utterance
