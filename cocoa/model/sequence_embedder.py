import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq import LuongAttention, BahdanauAttention, AttentionWrapper, AttentionWrapperState
from word_embedder import WordEmbedder
from rnn_cell import build_rnn_cell, MultiAttentionWrapper
from cocoa.model.util import transpose_first_two_dims
from itertools import izip

def add_sequence_embedder_arguments(parser):
    parser.add_argument('--rnn-size', type=int, default=20, help='Dimension of hidden units of RNN')
    parser.add_argument('--rnn-type', default='lstm', help='Type of RNN unit {rnn, gru, lstm}')
    parser.add_argument('--num-layers', type=int, default=1, help='Number of RNN layers')

def get_sequence_embedder(embedder_type, **kwargs):
    if embedder_type == 'bow':
        return BoWEmbedder(kwargs['vocab_size'], kwargs['embed_size'], kwargs.get('word_embedder', None))
    elif embedder_type == 'rnn':
        return RNNEmbedder(kwargs['embed_size'], kwargs['rnn_type'], kwargs['num_layers'], kwargs.get('aggregation', 'last'), kwargs['keep_prob'])
    elif embedder_type == 'rnn-attn':
        return AttentionRNNEmbedder(kwargs['embed_size'], kwargs.get('attn_size', None), kwargs['rnn_type'], kwargs['num_layers'], kwargs.get('aggregation', 'last'), kwargs['keep_prob'])
    raise ValueError('Unknown embedder_type %s' % embedder_type)

class SequenceEmbedder(object):
    '''
    Embed a sequence into a vector.
    NOTE: all inputs/outputs are time-major, i.e. (seq_len, batch_size, ...)
    '''
    def __init__(self, embed_size, aggregation):
        self.embed_size = embed_size
        self.aggregation = aggregation
        self.feedable_vars = {}

    def embed(self, sequence):
        raise NotImplementedError

    def build_seq_inputs(self, inputs, word_embedder, pad, time_major=False):
        '''
        inputs: a batch of input tokens/integers
        '''
        if not time_major:
            inputs = tf.transpose(inputs)
        mask = self.mask_paddings(inputs, pad)  # (seq_len, batch_size)
        inputs = word_embedder.embed(inputs)  # (seq_len, batch_size, embed_size)
        return inputs, mask

    @classmethod
    def concat_vector_to_seq(cls, context, sequence):
        '''
        context: (batch_size, context_size)
        sequence: (seq_len, batch_size, embed_size)
        return (seq_len, batch_size, embed_size+context_size)
        '''
        context = tf.to_float(context)
        context_seq = tf.tile(tf.expand_dims(context, 0), tf.stack([tf.shape(sequence)[0], 1, 1]))
        new_seq = tf.concat([sequence, context_seq], axis=2)
        return new_seq

    def mask_paddings(self, sequence, pad):
        '''
        Return a boolean tensor of (batch_size, seq_len) where padding positions are False.
        '''
        return tf.not_equal(sequence, pad)

    def zero_masked_embeddings(self, embeddings, mask):
        batch_size = tf.shape(embeddings)[1]
        embed_size = embeddings.get_shape().as_list()[-1]
        embeddings = tf.reshape(embeddings, [-1, embed_size])
        mask = tf.reshape(mask, [-1])
        masked_embeddings = tf.reshape(tf.where(mask, embeddings, tf.zeros_like(embeddings)), [-1, batch_size, embed_size])
        return masked_embeddings

    def sum(self, sequence, mask):
        '''
        sequence: (seq_len, batch_size, embed_size)
        '''
        sequence = self.zero_masked_embeddings(sequence, mask)
        return tf.reduce_sum(sequence, axis=0)

    def max(self, sequence, mask):
        '''
        sequence: (seq_len, batch_size, embed_size)
        '''
        return tf.reduce_max(sequence, axis=0)

    def aggregate(self, embeddings, mask=None):
        '''
        embeddings: (seq_len, batch_size, embed_size)
        mask: (seq_len, batch_size)
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
    def __init__(self, vocab_size=None, embed_size=None, word_embedder=None, aggregation='sum'):
        assert aggregation in ('sum', 'max', 'mean')
        super(BoWEmbedder, self).__init__(embed_size, aggregation)

        if word_embedder:
            self.vocab_size = word_embedder.num_symbols
            self.word_embedder = word_embedder
            self.embed_size = word_embedder.embed_size
        else:
            with tf.variable_scope(type(self).__name__):
                self.word_embedder = WordEmbedder(vocab_size, embed_size)

    def embed(self, sequence, padding_mask, **kwargs):
        '''
        sequence: (seq_len, batch_size). tf.int32.
        kwargs:
            integer: False means that the sequence is already embedded.
        '''
        with tf.variable_scope(type(self).__name__):
            if kwargs['integer']:
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
    def select_states(cls, states, mask):
        '''
        Return states at a specific time step.
        states: tensor or tuple of tensors from the scan output. (seq_len, batch_size, ...)
        mask: (seq_len, batch_size) where True indicates that state at that time step is to be returned.
        '''
        flat_states = nest.flatten(states)
        flat_selected_states = []
        mask = transpose_first_two_dims(mask)  # (batch_size, seq_len)
        for fs in flat_states:
            # NOTE: this is attention_state.time, which we are not using.
            # time assumes to be the same for all examples in a batch, which is not
            # compatible here because we take states at different time steps for
            # examples in the same batch (because they have different lengths).
            if len(fs.get_shape().as_list()) < 2:
                selected_states = fs[0]
            else:
                # fs is (seq_len, batch_size, ...)
                fs = transpose_first_two_dims(fs)  # (batch_size, seq_len, ...)
                # boolean_mask selects sub-tensors in the row order of mask
                # so we want each row to correspond to a sequence such that
                # for each batch, one element in the sequence is selected
                selected_states = tf.boolean_mask(fs, mask)  # (batch_size, ...)
            flat_selected_states.append(selected_states)
        selected_states = nest.pack_sequence_as(states, flat_selected_states)
        return selected_states

    def last(self, sequence, mask):
        '''
        sequence: a tuple of tensors (seq_len, batch_size, ...)
        mask: (seq_len, batch_size)
        '''
        # Index of the last non-masked entry
        last_inds = tf.reduce_sum(tf.where(mask, tf.ones_like(mask, dtype=tf.int32), tf.zeros_like(mask, dtype=tf.int32)), 0)  # (batch_size,)
        # For all-pad inputs
        last_inds = tf.where(tf.equal(last_inds, 0), tf.ones_like(last_inds), last_inds)
        # Index starts from 0
        last_inds -= 1

        seq_len = tf.shape(mask)[0]
        last_inds_mask = tf.cast(tf.one_hot(last_inds, seq_len, on_value=1, off_value=0), tf.bool)  # (batch_size, seq_len)
        last_inds_mask = tf.transpose(last_inds_mask)  # (seq_len, batch_size)
        return self.select_states(sequence, last_inds_mask)

    def aggregate(self, embeddings, mask=None):
        if self.aggregation == 'last':
            return self.last(embeddings, mask)
        return super(RNNEmbedder, self).aggregate(embeddings, mask)

    def _build_rnn_cell(self, input_size, **kwargs):
        return build_rnn_cell(self.rnn_type, self.embed_size, self.num_layers, self.keep_prob, input_size=input_size)

    def _build_init_state(self, cell, batch_size, init_cell_state=None):
        if init_cell_state is None:
            init_state = cell.zero_state(batch_size, tf.float32)
        else:
            init_state = init_cell_state
        self.feedable_vars['init_cell_state'] = init_state
        return init_state

    def embed(self, sequence, padding_mask, **kwargs):
        '''
        Assume the sequence is a tensor, i.e. tokens are already embedded.
        sequence: time-major (seq_len, batch_size, input_size)
        kwargs:
            init_state
        '''
        batch_size = tf.shape(sequence)[1]
        input_size = sequence.get_shape().as_list()[-1]
        with tf.variable_scope(type(self).__name__):
            cell = self._build_rnn_cell(input_size, **kwargs)

            init_state = self._build_init_state(cell, batch_size, init_cell_state=kwargs['init_cell_state'])
            self.feedable_vars['init_state'] = init_state

            init_output = tf.zeros([batch_size, cell.output_size])
            outputs, states = tf.scan(lambda a, x: cell(x, a[1]), sequence, initializer=(init_output, init_state), parallel_iterations=32)

            final_state = self.last(states, padding_mask)
            embedding = self.aggregate(outputs, padding_mask)
        # NOTE: outputs are all time-major
        return {'step_embeddings': outputs, 'embedding': embedding, 'final_state': final_state}

class AttentionRNNEmbedder(RNNEmbedder):
    def __init__(self, rnn_size, embed_size=None, rnn_type='lstm', num_layers=1, aggregation='last', keep_prob=1.):
        super(AttentionRNNEmbedder, self).__init__(rnn_size, rnn_type=rnn_type, num_layers=num_layers, aggregation=aggregation, keep_prob=keep_prob)
        self.embed_size = embed_size if embed_size is not None else rnn_size

    def _mask_to_memory_len(self, mask):
        '''
        mask: (batch_size, mem_len)
        '''
        if mask is not None:
            memory_len = tf.reduce_sum(tf.where(mask, tf.ones_like(mask, dtype=tf.int32), tf.zeros_like(mask, dtype=tf.int32)), 1)  # (batch_size,)
        else:
            memory_len = None
        return memory_len

    def _build_init_state(self, cell, batch_size, init_cell_state=None):
        zero_state = cell.zero_state(batch_size, dtype=tf.float32)
        if init_cell_state is None:
            init_state = zero_state
        # init_state can be from a RNN (no attention) encoder
        else:
            init_state = zero_state.clone(cell_state=init_cell_state)
        self.feedable_vars['init_cell_state'] = init_state.cell_state
        return init_state

    def _build_rnn_cell(self, input_size, **kwargs):
        attention_size = self.embed_size
        input_size += attention_size
        cell = super(AttentionRNNEmbedder, self)._build_rnn_cell(input_size, **kwargs)
        memory = kwargs['attention_memory']  # (batch_size, mem_len, mem_size)
        mask = kwargs.get('attention_mask', None)  # (batch_size, mem_len)
        #if not (isinstance(memory, list) or isinstance(memory, tuple)):
        if len(memory) == 1:
            memory = memory[0]
            memory_len = self._mask_to_memory_len(mask)
            attention_mechanism = LuongAttention(attention_size, memory, memory_sequence_length=memory_len, scale=True)
            cell = AttentionWrapper(cell, attention_mechanism, attention_layer_size=attention_size)
        else:
            if mask is not None:
                assert len(memory) == len(mask)
                memory_len = map(self._mask_to_memory_len, mask)
            else:
                memory_len = [None] * len(memory)
            attention_mechanisms = [LuongAttention(attention_size, m, l, scale=True) for m, l in izip(memory, memory_len)]
            attention_sizes = [attention_size for _ in attention_mechanisms]
            cell = AttentionWrapper(cell, attention_mechanisms, attention_layer_size=attention_sizes)
            # attention_size: project size of multiple context
            #cell = MultiAttentionWrapper(cell, attention_mechanisms, attention_size, attention_layer_size=attention_size)
        return cell

