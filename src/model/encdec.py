'''
NN models that take a sequence of words and actions.
Encode when action is read and decode when action is write.
'''

import tensorflow as tf
import numpy as np
from rnn_cell import AttnRNNCell, add_attention_arguments, build_rnn_cell
from graph import Graph, GraphMetadata
from graph_embedder import GraphEmbedder, GraphEmbedderConfig
from word_embedder import WordEmbedder
from vocab import is_entity
from preprocess import EOT
from util import transpose_first_two_dims, batch_linear
from tensorflow.python.util import nest
from itertools import izip

def add_model_arguments(parser):
    parser.add_argument('--model', default='encdec', help='Model name {encdec}')
    parser.add_argument('--rnn-size', type=int, default=20, help='Dimension of hidden units of RNN')
    parser.add_argument('--rnn-type', default='lstm', help='Type of RNN unit {rnn, gru, lstm}')
    parser.add_argument('--num-layers', type=int, default=1, help='Number of RNN layers')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of examples per batch')
    parser.add_argument('--word-embed-size', type=int, default=20, help='Word embedding size')
    add_attention_arguments(parser)

def build_model(schema, mappings, args):
    tf.reset_default_graph()
    tf.set_random_seed(args.random_seed)

    vocab = mappings['vocab']
    pad = vocab.to_ind(vocab.PAD)
    word_embedder = WordEmbedder(vocab.size, args.word_embed_size)
    if args.model == 'encdec':
        encoder = BasicEncoder(args.rnn_size, args.rnn_type, args.num_layers)
        decoder = BasicDecoder(args.rnn_size, vocab.size, args.rnn_type, args.num_layers)
        model = BasicEncoderDecoder(word_embedder, encoder, decoder, pad)
    elif args.model == 'attn-encdec' or args.model == 'attn-copy-encdec':
        max_degree = args.num_items + len(schema.attributes)
        graph_metadata = GraphMetadata(schema, mappings['entity'], mappings['relation'], args.rnn_size, args.max_num_entities, max_degree=max_degree, entity_hist_len=args.entity_hist_len, entity_cache_size=args.entity_cache_size)
        graph_embedder_config = GraphEmbedderConfig(args.node_embed_size, args.edge_embed_size, graph_metadata, entity_embed_size=args.entity_embed_size, use_entity_embedding=args.use_entity_embedding, mp_iters=args.mp_iters)
        Graph.metadata = graph_metadata
        graph_embedder = GraphEmbedder(graph_embedder_config)
        encoder = GraphEncoder(args.rnn_size, graph_embedder, rnn_type=args.rnn_type, num_layers=args.num_layers)
        if args.model == 'attn-encdec':
            decoder = GraphDecoder(args.rnn_size, vocab.size, graph_embedder, rnn_type=args.rnn_type, num_layers=args.num_layers)
        elif args.model == 'attn-copy-encdec':
            decoder = CopyGraphDecoder(args.rnn_size, vocab.size, graph_embedder, rnn_type=args.rnn_type, num_layers=args.num_layers)
        model = GraphEncoderDecoder(word_embedder, graph_embedder, encoder, decoder, pad)
    else:
        raise ValueError('Unknown model')
    return model

class BasicEncoder(object):
    '''
    A basic RNN encoder.
    '''
    def __init__(self, rnn_size, rnn_type='lstm', num_layers=1):
        self.rnn_size = rnn_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers

    def _build_init_output(self, cell):
        '''
        Initializer for scan. Should have the same shape as the RNN output.
        '''
        return tf.zeros([self.batch_size, cell.output_size])

    def _get_final_state(self, states):
        '''
        Return the final non-pad state from tf.scan outputs.
        '''
        with tf.name_scope(type(self).__name__+'/get_final_state'):
            flat_states = nest.flatten(states)
            #last_ind = tf.shape(flat_states[0])[0] - 1
            #flat_last_states = [tf.nn.embedding_lookup(state, last_ind) for state in flat_states]
            flat_last_states = []
            for state in flat_states:
                state = transpose_first_two_dims(state)  # (batch_size, time_seq, state_size)
                # For each batch, gather(state, last_ind). state: (time_seq, state_size)
                last_state = tf.map_fn(lambda x: tf.gather(x[0], x[1]),
                        (state, self.last_inds), dtype=state.dtype)  # (batch_size, state_size)
                flat_last_states.append(last_state)
            last_states = nest.pack_sequence_as(states, flat_last_states)
        return last_states

    def _build_rnn_cell(self):
        return build_rnn_cell(self.rnn_type, self.rnn_size, self.num_layers)

    def _build_init_state(self, cell, input_dict, initial_state):
        if initial_state is not None:
            return initial_state
        else:
            return cell.zero_state(self.batch_size, tf.float32)

    def build_model(self, input_dict, word_embedder, initial_state=None, time_major=True, scope=None):
        '''
        inputs: (batch_size, seq_len, input_size)
        '''
        with tf.variable_scope(scope or type(self).__name__):
            inputs = input_dict['inputs']
            self.batch_size = tf.shape(inputs)[0]

            cell = self._build_rnn_cell()
            self.init_state = self._build_init_state(cell, input_dict, initial_state)
            self.output_size = cell.output_size

            inputs = word_embedder.embed(inputs)
            if not time_major:
                inputs = transpose_first_two_dims(inputs)  # (seq_len, batch_size, input_size)

            rnn_outputs, states = tf.scan(lambda a, x: cell(x, a[1]), inputs, initializer=(self._build_init_output(cell), self.init_state))

            self.last_inds = tf.placeholder(tf.int32, shape=[None], name='last_inds')

        return self._build_output_dict(rnn_outputs, states)

    def _build_output_dict(self, rnn_outputs, rnn_states):
        final_state = self._get_final_state(rnn_states)
        return {'outputs': rnn_outputs, 'final_state': final_state}

class GraphEncoder(BasicEncoder):
    '''
    RNN encoder that update knowledge graph at the end.
    '''
    def __init__(self, rnn_size, graph_embedder, rnn_type='lstm', num_layers=1):
        super(GraphEncoder, self).__init__(rnn_size, rnn_type, num_layers)
        self.graph_embedder = graph_embedder

    def build_model(self, input_dict, word_embedder, initial_state=None, time_major=True, scope=None):
        output_dict = super(GraphEncoder, self).build_model(input_dict, word_embedder, initial_state=initial_state, time_major=time_major, scope=scope)

        # Use the final encoder state as the utterance embedding
        final_output = self._get_final_state(output_dict['outputs'])
        new_utterances = self.graph_embedder.update_utterance(input_dict['entities'], final_output, input_dict['utterances'])
        with tf.variable_scope(tf.get_variable_scope(), reuse=self.graph_embedder.context_initialized):
            context = self.graph_embedder.get_context(new_utterances)
        output_dict['utterances'] = new_utterances
        output_dict['context'] = context
        return output_dict

class BasicDecoder(BasicEncoder):
    def __init__(self, rnn_size, num_symbols, rnn_type='lstm', num_layers=1):
        super(BasicDecoder, self).__init__(rnn_size, rnn_type, num_layers)
        self.num_symbols = num_symbols

    def _build_output(self, output_dict):
        '''
        Take RNN outputs and produce logits over the vocab.
        '''
        outputs = output_dict['outputs']
        outputs = transpose_first_two_dims(outputs)  # (batch_size, seq_len, output_size)
        logits = batch_linear(outputs, self.num_symbols, True)
        return logits

    def build_model(self, input_dict, word_embedder, initial_state=None, time_major=True, scope=None):
        output_dict = super(BasicDecoder, self).build_model(input_dict, word_embedder, initial_state=initial_state, time_major=time_major, scope=scope)  # outputs: (seq_len, batch_size, output_size)
        with tf.variable_scope(scope or type(self).__name__):
            logits = self._build_output(output_dict)
        output_dict['logits'] = logits
        return output_dict

class GraphDecoder(GraphEncoder):
    '''
    Decoder with attention mechanism over the graph.
    '''
    # TODO: group input args: rnn_config, attention_config etc.
    def __init__(self, rnn_size, num_symbols, graph_embedder, rnn_type='lstm', num_layers=1, scoring='linear', output='project'):
        super(GraphDecoder, self).__init__(rnn_size, graph_embedder, rnn_type, num_layers)
        self.num_symbols = num_symbols
        # Config for the attention cell
        self.context_size = self.graph_embedder.config.context_size
        self.scorer = scoring
        self.output_combiner = output

    def _build_rnn_cell(self):
        return AttnRNNCell(self.rnn_size, self.context_size, self.rnn_type, self.scorer, self.output_combiner, self.num_layers)

    def _build_init_output(self, cell):
        '''
        Output includes both RNN output and attention scores.
        '''
        output = super(GraphDecoder, self)._build_init_output(cell)
        return (output, tf.zeros_like(self.graph_embedder.node_ids, dtype=tf.float32))

    def _build_output(self, output_dict):
        '''
        Take RNN outputs and produce logits over the vocab.
        '''
        outputs = output_dict['outputs']
        outputs = transpose_first_two_dims(outputs)  # (batch_size, seq_len, output_size)
        logits = batch_linear(outputs, self.num_symbols, True)
        return logits

    def _build_init_state(self, cell, input_dict, initial_state):
        if initial_state is not None:
            # NOTE: we assume that the initial state comes from the encoder and is just
            # the rnn state. We need to compute attention and get context for the attention
            # cell's initial state.
            return cell.init_state(initial_state, input_dict['init_output'], input_dict['context'])
        else:
            return cell.zero_state(self.batch_size, input_dict['context'])

    def build_model(self, input_dict, word_embedder, initial_state=None, time_major=True, scope=None):
        output_dict = super(GraphDecoder, self).build_model(input_dict, word_embedder, initial_state=initial_state, time_major=time_major, scope=scope)  # outputs: (seq_len, batch_size, output_size)
        with tf.variable_scope(scope or type(self).__name__):
            logits = self._build_output(output_dict)
        output_dict['logits'] = logits
        return output_dict

    def _build_output_dict(self, rnn_outputs, rnn_states):
        final_state = self._get_final_state(rnn_states)
        outputs, attn_scores = rnn_outputs
        return {'outputs': outputs, 'attn_scores': attn_scores, 'final_state': final_state}

class CopyGraphDecoder(GraphDecoder):
    '''
    Decoder with copy mechanism over the attention context.
    '''
    def _build_output(self, output_dict):
        '''
        Take RNN outputs and produce logits over the vocab and the attentions.
        '''
        logits = super(CopyGraphDecoder, self)._build_output(output_dict)  # (batch_size, seq_len, num_symbols)
        attn_scores = transpose_first_two_dims(output_dict['attn_scores'])
        return tf.concat(2, [logits, attn_scores])


class BasicEncoderDecoder(object):
    '''
    Basic seq2seq model.
    '''
    def __init__(self, word_embedder, encoder, decoder, pad, scope=None):
        self.PAD = pad  # Id of PAD in the vocab
        self.encoder = encoder
        self.decoder = decoder
        self.build_model(word_embedder, encoder, decoder, scope)

    def update_feed_dict(self, **kwargs):
        feed_dict = kwargs.pop('feed_dict', {})
        for key, val in kwargs.iteritems():
            if val is not None:
                feed_dict[getattr(self, key)] = val
        return feed_dict

    def compute_loss(self, logits, targets):
        '''
        logits: (batch_size, seq_len, vocab_size)
        targets: (batch_size, seq_len)
        '''
        shape = tf.shape(logits)
        batch_size = shape[0]
        num_symbols = shape[2]
        # sparse_softmax_cross_entropy_with_logits only takes 2D tensors
        logits = tf.reshape(logits, [-1, num_symbols])
        targets = tf.reshape(targets, [-1])
        # Mask padded tokens
        token_weights = tf.cast(tf.not_equal(targets, tf.constant(self.PAD)), tf.float32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets) * token_weights
        token_weights = tf.reduce_sum(tf.reshape(token_weights, [batch_size, -1]), 1) + 1e-12
        # Average over words in each sequence
        loss = tf.reduce_sum(tf.reshape(loss, [batch_size, -1]), 1) / token_weights
        # Average over sequences in the batch
        loss = tf.reduce_sum(loss, 0) / tf.to_float(batch_size)
        return loss

    def _encoder_input_dict(self):
        return {
                'inputs': tf.placeholder(tf.int32, shape=[None, None], name='encoder_inputs'),
               }

    def _decoder_input_dict(self, encoder_output_dict):
        return {
                'inputs': tf.placeholder(tf.int32, shape=[None, None], name='decoder_inputs'),
               }

    def _decoder_initial_state(self, encoder_output_dict):
        return encoder_output_dict['final_state']

    def build_model(self, word_embedder, encoder, decoder, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # Encoding
            with tf.name_scope('Encoder'):
                encoder_input_dict = self._encoder_input_dict()
                encoder_output_dict = encoder.build_model(encoder_input_dict, word_embedder, time_major=False)

            # Decoding
            with tf.name_scope('Decoder'):
                decoder_input_dict = self._decoder_input_dict(encoder_output_dict)
                # TODO: don't need a function here
                decoder_initial_state = self._decoder_initial_state(encoder_output_dict)
                decoder_output_dict = decoder.build_model(decoder_input_dict, word_embedder, initial_state=decoder_initial_state, time_major=False)

            # Placehoders (feed in through feed_dict)
            self.encoder_inputs = encoder_input_dict['inputs']
            self.decoder_inputs = decoder_input_dict['inputs']
            self.encoder_inputs_last_inds = encoder.last_inds
            self.decoder_inputs_last_inds = decoder.last_inds
            self.targets = tf.placeholder(tf.int32, shape=[None, None], name='targets')

            # Outputs (accessed through sess.run)
            self.encoder_final_state = encoder_output_dict['final_state']
            self.decoder_final_state = decoder_output_dict['final_state']
            self.decoder_outputs = decoder_output_dict['outputs']
            # Loss
            self.logits = decoder_output_dict['logits']
            self.loss = self.compute_loss(self.logits, self.targets)

            # Feedable tensors (feed in through feed_dict)
            self.encoder_init_state = encoder.init_state
            self.decoder_init_state = decoder.init_state

        return encoder_input_dict, encoder_output_dict, decoder_input_dict, decoder_output_dict

    @classmethod
    def get_prediction(cls, logits):
        '''
        Return predicted vocab from output/logits (batch_size, seq_len, vocab_size).
        '''
        preds = np.argmax(logits, axis=2)  # (batch_size, seq_len)
        return preds

    # TODO: put this in evaluator
    def generate(self, sess, batch, encoder_init_state, max_len, copy=False, vocab=None, graphs=None, utterances=None):
        if copy:
            encoder_inputs = graphs.entity_to_vocab(batch['encoder_inputs'], vocab)
            decoder_inputs = graphs.entity_to_vocab(batch['decoder_inputs'], vocab)
        else:
            encoder_inputs = batch['encoder_inputs']
            decoder_inputs = batch['decoder_inputs']
        batch_size = encoder_inputs.shape[0]

        # Initial feed dict, with encoder inputs, and <go> to start the decoder
        decoder_inputs_last_inds = np.zeros_like(batch['decoder_inputs_last_inds'], dtype=np.int32)
        feed_dict = self.update_feed_dict(encoder_inputs=encoder_inputs,
                                          decoder_inputs=decoder_inputs[:, [0]],
                                          encoder_inputs_last_inds=batch['encoder_inputs_last_inds'],
                                          decoder_inputs_last_inds=decoder_inputs_last_inds,
                                          encoder_init_state=encoder_init_state)
        if graphs is not None:
            graph_data = graphs.get_batch_data(batch['encoder_tokens'], None, utterances)
            feed_dict = self.update_feed_dict(feed_dict=feed_dict,
                    encoder_entities=graph_data['encoder_entities'],
                    encoder_input_utterances=graph_data['utterances'])
            self.add_graph_data(feed_dict, graph_data)

        preds = np.zeros([batch_size, max_len], dtype=np.int32)
        true_final_state = None
        for i in xrange(max_len):
            logits, final_state = sess.run([self.logits, self.decoder_final_state], feed_dict=feed_dict)
            # After step 0, we will use our prediction as input instead of the ground true
            if i == 0:
                true_final_state = final_state
            step_decoder_inputs = self.get_prediction(logits)
            preds[:, [i]] = step_decoder_inputs
            if copy:
                # Convert local node ids to global entity ids
                step_decoder_inputs = graphs.copy_preds(step_decoder_inputs, vocab.size)
                # Convert entity ids to vocab ids
                step_decoder_inputs = graphs.entity_to_vocab(step_decoder_inputs, vocab)
            feed_dict = self.update_feed_dict(decoder_inputs=step_decoder_inputs,
                                              decoder_inputs_last_inds=decoder_inputs_last_inds,
                                              decoder_init_state=final_state)

        # Decode over the ground truth
        # -1 because we start from the second input token
        decoder_inputs_last_inds = batch['decoder_inputs_last_inds'] - 1
        # NOTE: we get <0 indices when this is a padded turn, i.e. <pad> <pad> <pad> ...
        # At real test time, we want to use batch_size = 1 so this shouldn't happen
        decoder_inputs_last_inds[decoder_inputs_last_inds < 0] = 0
        feed_dict = self.update_feed_dict(decoder_inputs=decoder_inputs[:, 1:],
                                          decoder_inputs_last_inds=decoder_inputs_last_inds,
                                          decoder_init_state=true_final_state)
        if graphs is not None:
            # Since the graph structure may change during decoding, we cannot continue from
            # the encoder's state. So we just run it all over again.
            # Since the graph has been updated by encoder_tokens, here we only update it using
            # the decoder_tokens. The utterances is the utterances before runing through the
            # encoder, and it is resized according to the new graph structure if necessary.
            new_graph_data = graphs.get_batch_data(None, batch['decoder_tokens'], utterances)
            feed_dict = self.update_feed_dict(encoder_inputs=encoder_inputs,
                    decoder_inputs=decoder_inputs,
                    encoder_inputs_last_inds=batch['encoder_inputs_last_inds'],
                    decoder_inputs_last_inds=batch['decoder_inputs_last_inds'],
                    encoder_init_state=encoder_init_state,
                    encoder_entities=graph_data['encoder_entities'],
                    decoder_entities=new_graph_data['decoder_entities'],
                    encoder_input_utterances=new_graph_data['utterances'])
            self.add_graph_data(feed_dict, graph_data)
            [true_final_state, utterances] = sess.run([self.decoder_final_state, self.decoder_output_utterances], feed_dict=feed_dict)
        else:
            [true_final_state] = sess.run([self.decoder_final_state], feed_dict=feed_dict)
        return preds, final_state, true_final_state, utterances

class GraphEncoderDecoder(BasicEncoderDecoder):
    def __init__(self, word_embedder, graph_embedder, encoder, decoder, pad, scope=None):
        self.graph_embedder = graph_embedder
        super(GraphEncoderDecoder, self).__init__(word_embedder, encoder, decoder, pad, scope)

    def _encoder_input_dict(self):
        with tf.name_scope(type(self).__name__+'/encoder_input_dict'):
            input_dict = super(GraphEncoderDecoder, self)._encoder_input_dict()
            input_dict['utterances'] = tf.placeholder(tf.float32, shape=[None, None, self.graph_embedder.config.utterance_size], name='utterances')
            input_dict['entities'] = tf.placeholder(tf.int32, shape=[None, self.graph_embedder.config.entity_cache_size], name='encoder_entities')
        return input_dict

    def _decoder_input_dict(self, encoder_output_dict):
        with tf.name_scope(type(self).__name__+'/decoder_input_dict'):
            input_dict = super(GraphEncoderDecoder, self)._decoder_input_dict(encoder_output_dict)
            # This is used to compute the initial attention
            input_dict['init_output'] = self.encoder._get_final_state(encoder_output_dict['outputs'])
            input_dict['utterances'] = encoder_output_dict['utterances']
            input_dict['entities'] = tf.placeholder(tf.int32, shape=[None, self.graph_embedder.config.entity_cache_size], name='decoder_entities')
            input_dict['context'] = encoder_output_dict['context']
        return input_dict

    def build_model(self, word_embedder, encoder, decoder, scope=None):
        encoder_input_dict, encoder_output_dict, decoder_input_dict, decoder_output_dict = super(GraphEncoderDecoder, self).build_model(word_embedder, encoder, decoder, scope)
        # TODO: should probably group the feedable inputs and runnable outputs
        with tf.variable_scope(scope or type(self).__name__):
            # Placeholders
            self.encoder_entities = encoder_input_dict['entities']
            self.decoder_entities = decoder_input_dict['entities']
            self.encoder_input_utterances = encoder_input_dict['utterances']
            self.decoder_input_utterances = decoder_input_dict['utterances']
            self.graph_structure = self.graph_embedder.input_data
            # Outputs: updated utterances and context
            self.encoder_output_utterances = encoder_output_dict['utterances']
            self.decoder_output_utterances = decoder_output_dict['utterances']
            self.encoder_output_context = encoder_output_dict['context']
            self.decoder_output_context = decoder_output_dict['context']

    def add_graph_data(self, feed_dict, graph_data):
        '''
        Add graph-related data to feed_dict.
        '''
        feed_dict = self.update_feed_dict(feed_dict=feed_dict,
                graph_structure=(graph_data['node_ids'],
                    graph_data['mask'],
                    graph_data['entity_ids'],
                    graph_data['paths'],
                    graph_data['node_paths'],
                    graph_data['node_feats']))
        return feed_dict

