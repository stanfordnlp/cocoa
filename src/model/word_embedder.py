import tensorflow as tf

class WordEmbedder(object):
    def __init__(self, num_symbols, embed_size, pad=None, scope=None):
        self.num_symbols = num_symbols
        self.embed_size = embed_size
        self.pad = pad
        self.build_model(scope)

    def build_model(self, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            self.embedding = tf.get_variable('embedding', [self.num_symbols, self.embed_size])

    def embed(self, inputs, zero_pad=False):
        embeddings = tf.nn.embedding_lookup(self.embedding, inputs)
        if self.pad is not None and zero_pad:
            embeddings = tf.where(inputs == self.pad, tf.zeros_like(embeddings), embeddings)
        return embeddings
