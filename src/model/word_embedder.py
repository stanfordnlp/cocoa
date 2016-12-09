import tensorflow as tf

class WordEmbedder(object):
    def __init__(self, num_symbols, embed_size, scope=None):
        self.num_symbols = num_symbols
        self.embed_size = embed_size
        self.build_model(scope)

    def build_model(self, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            self.embedding = tf.get_variable('embedding', [self.num_symbols, self.embed_size])

    def embed(self, inputs, pad=None):
        embeddings = tf.nn.embedding_lookup(self.embedding, inputs)
        if pad is not None:
            embeddings = tf.select(inputs == pad, tf.zeros_like(embeddings), embeddings)
        return embeddings
