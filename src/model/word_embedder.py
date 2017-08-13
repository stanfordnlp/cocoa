import tensorflow as tf

# TODO: add dropout
class WordEmbedder(object):
    def __init__(self, num_symbols, embed_size, pretrained_embeddings=None, pad=None, scope=None):
        self.num_symbols = num_symbols
        self.embed_size = embed_size
        self.pad = pad
        self.build_model(pretrained_embeddings, scope)

    def build_model(self, pretrained_embeddings=None, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            if pretrained_embeddings is None:
                self.embedding = tf.get_variable('embedding', [self.num_symbols, self.embed_size])
            else:
                self.embed_size = pretrained_embeddings.shape[1]
                initializer = tf.constant_initializer(pretrained_embeddings)
                self.embedding = tf.get_variable('embedding', [self.num_symbols, self.embed_size], initializer=initializer)

    def embed(self, inputs, zero_pad=False):
        embeddings = tf.nn.embedding_lookup(self.embedding, inputs)
        if self.pad is not None and zero_pad:
            embeddings = tf.where(inputs == self.pad, tf.zeros_like(embeddings), embeddings)
        return embeddings
