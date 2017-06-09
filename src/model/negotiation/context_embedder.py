import tensorflow as tf

class ContextEmbedder(object):
    def __init__(self, category_size):
        self.category_size = category_size
        self._build_inputs()

    def _build_inputs(self):
        with tf.variable_scope(type(self).__name__):
            self.category = tf.placeholder(tf.int32, shape=[None], name='category')  # (batch_size,)
            self.title = tf.placeholder(tf.int32, shape=[None, None], name='title')  # (batch_size, title_len)
            self.description = tf.placeholder(tf.int32, shape=[None, None], name='description')  # (batch_size, description_len)

    def one_hot_embed(self, inputs, size):
        return tf.one_hot(inputs, size, on_value=1, off_value=0)

    def embed(self):
        return self.one_hot_embed(self.category, self.category_size)

    def get_feed_dict(self, **kwargs):
        # TODO: add args to specify which info to include
        feed_dict = kwargs.pop('feed_dict', {})
        feed_dict[self.category] = kwargs.pop('category')
        return feed_dict
