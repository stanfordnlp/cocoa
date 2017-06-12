import tensorflow as tf

def add_context_embedder_arguments(parser):
    parser.add_argument('--context-encoder', choices=['bow', 'rnn'], default='bow', help='Encoding of context')
    parser.add_argument('--context-size', type=int, default=20, help='Embedding size of context')
    parser.add_argument('--context', nargs='*', help='What context to use (title, description, category)')

class ContextEmbedder(object):
    def __init__(self, category_size, word_embedder, seq_embedder, pad):
        self.category_size = category_size
        self.word_embedder = word_embedder
        self.seq_embedder = seq_embedder
        self.pad = pad
        self._build_inputs()

    def _build_inputs(self):
        with tf.variable_scope(type(self).__name__):
            self.category = tf.placeholder(tf.int32, shape=[None], name='category')  # (batch_size,)
            self.title = tf.placeholder(tf.int32, shape=[None, None], name='title')  # (batch_size, title_len)
            self.description = tf.placeholder(tf.int32, shape=[None, None], name='description')  # (batch_size, description_len)

    def one_hot_embed(self, inputs, size):
        return tf.one_hot(inputs, size, on_value=1, off_value=0)

    def _build_seq_inputs(self, inputs):
        return self.word_embedder.embed(tf.transpose(inputs))

    def _embed_seq(self, sequence):
        inputs, mask = self.seq_embedder.build_seq_inputs(sequence, self.word_embedder, self.pad, time_major=False)
        embeddings = self.seq_embedder.embed(inputs, mask, integer=False, init_state=None)
        return embeddings['embedding']

    def embed(self, context=('category',)):
        category_embedding = tf.to_float(self.one_hot_embed(self.category, self.category_size))
        title_embedding = self._embed_seq(self.title)
        description_embedding = self._embed_seq(self.description)
        # embeddings: (batch_size, embed_size)
        embeddings = {
                'category': category_embedding,
                'title': title_embedding,
                'description': description_embedding,
                }
        return tf.concat([embeddings[k] for k in context], axis=1)

    def get_feed_dict(self, **kwargs):
        feed_dict = kwargs.pop('feed_dict', {})
        feed_dict[self.category] = kwargs.pop('category')
        feed_dict[self.title] = kwargs.pop('title')
        feed_dict[self.description] = kwargs.pop('description')
        return feed_dict
