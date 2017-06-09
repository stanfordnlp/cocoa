import tensorflow as tf
from word_embedder import WordEmbedder
from rnn_cell import build_rnn_cell

class SequenceEmbedder(object):
    def __init__(self, embed_size, pad, aggregation):
        self.embed_size = embed_size
        self.pad = pad
        self.aggregation = aggregation

    def embed(self, sequence):
        raise NotImplementedError

    def mask_paddings(self, sequence):
        '''
        Return a boolean tensor of (batch_size, seq_len) where padding positions are False.
        '''
        if self.pad is None:
            raise ValueError('No padding value provided')
        return tf.not_equal(sequence, self.pad)

    def zero_masked_embeddings(self, embeddings, mask):
        batch_size = tf.shape(embeddings)[0]
        embeddings = tf.reshape(embeddings, [-1, self.embed_size])
        mask = tf.reshape(mask, [-1])
        masked_embeddings = tf.reshape(tf.where(mask, embeddings, tf.zeros_like(embeddings)), [batch_size, -1, self.embed_size])
        return masked_embeddings

    def aggregate(self, embeddings, mask=None):
        '''
        embeddings: (batch_size, seq_len, embed_size)
        mask: (batch_size, seq_len)
        return: (batch_size, embed_size)
        '''
        if mask is None:
            mask = tf.ones_like(embeddings[:, :], dtype=tf.bool)
        if self.aggregation == 'sum':
            embeddings = zero_masked_embeddings(embeddings, mask)
            summed_embedding = tf.reduce_sum(embeddings, axis=1)
            return summed_embedding
        elif self.aggregation == 'max':
            return tf.reduce_max(embeddings, axis=1)
        raise ValueError('Unknown aggregation')

class BoWEmbedder(SequenceEmbedder):
    def __init__(self, vocab_size, pad=None, embed_size=None, word_embedder=None, aggregation='sum'):
        assert aggregation in ('sum', 'max', 'mean')
        super(BoWEmbedder, self).__init__(embed_size, pad, aggregation)

        if word_embedder:
            self.word_embedder = word_embedder
            self.embed_size = word_embedder.embed_size
        else:
            with tf.variable_scope(type(self).__name__):
                self.word_embedder = WordEmbedder(vocab_size, embed_size, pad)

    def embed(self, sequence):
        '''
        sequence: (batch_size, seq_len). tf.int32.
        '''
        with tf.variable_scope(type(self).__name__):
            word_embeddings = self.word_embedder.embed(sequence)
            padding_mask = self.mask_paddings(sequence)
            embedding = self.aggregate(word_embeddings, padding_mask)
        return {'step_embeddings': word_embeddings, 'embedding': embedding}

class RNNEmbedder(SequenceEmbedder):
    def __init__(self, rnn_size, rnn_type='lstm', num_layers=1, pad=None, aggregation='last', keep_prob=1.):
        assert aggregation in ('sum', 'max', 'mean', 'last')
        super(RNNEmbedder, self).__init__(rnn_size, pad, aggregation)
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.keep_prob = keep_prob

    def embed(self, sequence, init_state=None):
        '''
        Assume the sequence is a tensor, i.e. tokens are already embedded.
        sequence: (batch_size, seq_len, input_size)
        '''
        with tf.variable_scope(type(self).__name__):
            cell = build_rnn_cell

