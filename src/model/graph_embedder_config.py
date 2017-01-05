class GraphEmbedderConfig(object):
    def __init__(self, node_embed_size, edge_embed_size, graph_metadata, entity_embed_size=None, use_entity_embedding=False, mp_iters=2, decay=1, msg_agg='sum', learned_decay=False):
        self.node_embed_size = node_embed_size

        self.num_edge_labels = graph_metadata.relation_map.size
        self.edge_embed_size = edge_embed_size

        # RNN output size
        self.utterance_size = graph_metadata.utterance_size
        self.decay = decay
        self.learned_decay = learned_decay

        # Size of input features from Graph
        self.feat_size = graph_metadata.feat_size

        # Number of message passing iterations
        self.mp_iters = mp_iters
        self.msg_agg = msg_agg

        self.context_size = self.node_embed_size * mp_iters
        # x2 because we encoder and decoder utterances are concatenated
        self.context_size += (self.utterance_size * 2 + self.feat_size)
        if use_entity_embedding:
            self.context_size += entity_embed_size

        self.use_entity_embedding = use_entity_embedding
        if use_entity_embedding:
            self.num_entities = graph_metadata.entity_map.size
            self.entity_embed_size = entity_embed_size

        # padding
        self.pad_path_id = graph_metadata.PAD_PATH_ID
        self.node_pad = graph_metadata.NODE_PAD

