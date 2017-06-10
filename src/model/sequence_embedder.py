import tensorflow as tf
from tensorflow.python.util import nest
from word_embedder import WordEmbedder
from rnn_cell import build_rnn_cell
from src.model.util import transpose_first_two_dims

def add_sequence_embedder_arguments(parser):
    parser.add_argument('--rnn-size', type=int, default=20, help='Dimension of hidden units of RNN')
    parser.add_argument('--rnn-type', default='lstm', help='Type of RNN unit {rnn, gru, lstm}')
    parser.add_argument('--num-layers', type=int, default=1, help='Number of RNN layers')

def get_sequence_embedder(embedder_type, **kwargs):
    if embedder_type == 'bow':
        return BoWEmbedder(kwargs['vocab_size'], kwargs['word_embedder'])
    elif embedder_type == 'rnn':
        return RNNEmbedder(kwargs['rnn_size'], kwargs['rnn_type'], kwargs['num_layers'], kwargs.get('aggregation', 'last'), kwargs['keep_prob'])
    raise ValueError('Unknown embedder_type %s' % embedder_type)

class SequenceEmbedder(object):
    def __init__(self, embed_size, aggregation):
        self.embed_size = embed_size
        self.aggregation = aggregation

    def embed(self, sequence):
        raise NotImplementedError

    def mask_paddings(self, sequence, pad):
        '''
        Return a boolean tensor of (batch_size, seq_len) where padding positions are False.
        '''
        return tf.not_equal(sequence, pad)

    def zero_masked_embeddings(self, embeddings, mask):
        batch_size = tf.shape(embeddings)[0]
        embeddings = tf.reshape(embeddings, [-1, self.embed_size])
        mask = tf.reshape(mask, [-1])
        masked_embeddings = tf.reshape(tf.where(mask, embeddings, tf.zeros_like(embeddings)), [batch_size, -1, self.embed_size])
        return masked_embeddings

    def sum(self, sequence, mask):
        '''
        sequence: (batch_size, seq_len, embed_size)
        '''
        sequence = self.zero_masked_embeddings(sequence, mask)
        return tf.reduce_sum(sequence, axis=1)

    def max(self, sequence, mask):
        '''
        sequence: (batch_size, seq_len, embed_size)
        '''
        return tf.reduce_max(sequence, axis=1)

    def aggregate(self, embeddings, mask=None):
        '''
        embeddings: (batch_size, seq_len, embed_size)
        mask: (batch_size, seq_len)
        return: (batch_size, embed_size)
        '''
        if mask is None:
            mask = tf.ones_like(embeddings[:, :], dtype=tf.bool)
        if self.aggregation == 'sum':
            return self.sum(embeddings, mask)
        elif self.aggregation == 'max':
            return self.max(embeddings, mask)
        raise ValueError('Unknown aggregation')

class BoWEmbedder(SequenceEmbedder):
    def __init__(self, vocab_size, embed_size=None, word_embedder=None, aggregation='sum'):
        assert aggregation in ('sum', 'max', 'mean')
        super(BoWEmbedder, self).__init__(embed_size, aggregation)

        if word_embedder:
            self.word_embedder = word_embedder
            self.embed_size = word_embedder.embed_size
        else:
            with tf.variable_scope(type(self).__name__):
                self.word_embedder = WordEmbedder(vocab_size, embed_size)

    def embed(self, sequence, padding_mask, integer=True):
        '''
        sequence: (batch_size, seq_len). tf.int32.
        integer is False means that the sequence is already embedded.
        '''
        with tf.variable_scope(type(self).__name__):
            if integer:
                word_embeddings = self.word_embedder.embed(sequence)
            else:
                word_embeddings = sequence
            embedding = self.aggregate(word_embeddings, padding_mask)
        return {'step_embeddings': word_embeddings, 'embedding': embedding}

class RNNEmbedder(SequenceEmbedder):
    def __init__(self, rnn_size, rnn_type='lstm', num_layers=1, aggregation='last', keep_prob=1.):
        assert aggregation in ('sum', 'max', 'mean', 'last')
        super(RNNEmbedder, self).__init__(rnn_size, aggregation)
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.keep_prob = keep_prob

    @classmethod
    def select_states(states, mask):
        '''
        Return states at a specific time step.
        states: tensor or tuple of tensors from the scan output. (seq_len, batch_size, ...)
        mask: (batch_size, seq_len) where True indicates that state at that time step is to be returned.
        '''
        flat_states = nest.flatten(states)
        flat_selected_states = []
        for states in flat_states:
            states = transpose_first_two_dims(states)  # (batch_size, seq_len, ...)
            selected_states = tf.boolean_mask(states, mask)
            flat_selected_states.append(selected_states)
        selected_states = nest.pack_sequence_as(states, flat_selected_states)
        return selected_states

    def last(self, sequence, mask):
        '''
        sequence: (batch_size, seq_len, embed_size)
        '''
        last_inds = tf.reduce_sum(tf.where(mask, tf.ones_like(sequence), tf.zeros_like(sequence)), 1)  # (batch_size,)
        last_inds -= 1  # Index starts from 0
        seq_len = tf.shape(sequence)[1]
        last_inds_mask = tf.cast(tf.one_hot(last_inds, seq_len, on_value=1, off_value=0), tf.bool)
        return self.select_states(sequence, last_inds_mask)

    def aggregate(self, embeddings, mask=None):
        if self.aggregation == 'last':
            return self.last(embeddings, mask)
        return super(RNNEmbedder, self).aggregate(embeddings, mask)

    def embed(self, sequence, padding_mask, init_state=None):
        '''
        Assume the sequence is a tensor, i.e. tokens are already embedded.
        sequence: (batch_size, seq_len, input_size)
        '''
        batch_size = tf.shape(sequence)[0]
        with tf.variable_scope(type(self).__name__):
            cell = build_rnn_cell(self.rnn_type, self.embed_size, self.num_layers, self.keep_prob)
            init_state = init_state or cell.zero_state(batch_size, tf.float32)
            init_output = tf.zeros([batch_size, cell.output_size])
            outputs, states = tf.scan(lambda a, x: cell(x, a[1]), sequence, initializer=(init_output, init_state))
            final_state = self.last(states, padding_mask)
            embedding = self.aggregate(outputs, padding_mask)
        return {'step_embeddings': outputs, 'embedding': embedding, 'final_state': final_state}

