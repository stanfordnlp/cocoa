'''
NN models that take a sequence of words and actions.
Encode when action is read and decode when action is write.
'''

import tensorflow as tf
import numpy as np
from rnn_cell import AttnRNNCell, add_attention_arguments, build_rnn_cell
from graph import Graph
from graph_embedder import GraphEmbedder
from word_embedder import WordEmbedder
from vocab import is_entity
from preprocess import EOT
from util import transpose_first_two_dims, batch_linear
from tensorflow.python.util import nest
from itertools import izip

def add_model_arguments(parser):
    parser.add_argument('--model', default='encdec', help='Model name {encdec}')
    parser.add_argument('--rnn-size', type=int, default=50, help='Dimension of hidden units of RNN')
    parser.add_argument('--rnn-type', default='lstm', help='Type of RNN unit {rnn, gru, lstm}')
    parser.add_argument('--num-layers', type=int, default=1, help='Number of RNN layers')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of examples per batch')
    parser.add_argument('--word-embed-size', type=int, default=50, help='Word embedding size')
    add_attention_arguments(parser)

class BasicEncoder(object):
    '''
    A basic RNN encoder.
    '''
    def __init__(self, rnn_size, rnn_type='lstm', num_layers=1, batch_size=1):
        self.rnn_size = rnn_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.batch_size = batch_size

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
            cell = self._build_rnn_cell()
            self.init_state = self._build_init_state(cell, input_dict, initial_state)
            self.output_size = cell.output_size

            inputs = input_dict['inputs']
            inputs = word_embedder.embed(inputs)
            if not time_major:
                inputs = transpose_first_two_dims(inputs)  # (seq_len, batch_size, input_size)

            rnn_outputs, states = tf.scan(lambda a, x: cell(x, a[1]), inputs, initializer=(self._build_init_output(cell), self.init_state))

            self.last_inds = tf.placeholder(tf.int32, shape=[self.batch_size], name='last_inds')

        return self._build_output_dict(rnn_outputs, states)

    def _build_output_dict(self, rnn_outputs, rnn_states):
        final_state = self._get_final_state(rnn_states)
        return {'outputs': rnn_outputs, 'final_state': final_state}

class GraphEncoder(BasicEncoder):
    '''
    RNN encoder that update knowledge graph at the end.
    '''
    def __init__(self, rnn_size, graph_embedder, rnn_type='lstm', num_layers=1, batch_size=1):
        super(GraphEncoder, self).__init__(rnn_size, rnn_type, num_layers, batch_size)
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
    def __init__(self, rnn_size, num_symbols, rnn_type='lstm', num_layers=1, batch_size=1):
        super(BasicDecoder, self).__init__(rnn_size, rnn_type, num_layers, batch_size)
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
    def __init__(self, rnn_size, num_symbols, graph_embedder, rnn_type='lstm', num_layers=1, batch_size=1, scoring='linear', output='project'):
        super(GraphDecoder, self).__init__(rnn_size, graph_embedder, rnn_type, num_layers, batch_size)
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
        # TODO: to have encoder continue from decoder's final state, need to decompose it..
        # currently it's state, attention, context
        return {'outputs': outputs, 'attn_scores': attn_scores, 'final_state': final_state}

class CopyGraphDecoder(GraphDecoder):
    pass

class BasicEncoderDecoder(object):
    '''
    Basic seq2seq model.
    '''
    def __init__(self, word_embedder, encoder, decoder, pad, scope=None):
        assert encoder.batch_size == decoder.batch_size
        self.batch_size = encoder.batch_size
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
        batch_size, _, num_symbols = logits.get_shape().as_list()
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
        loss = tf.reduce_sum(loss, 0) / batch_size
        return loss

    def _encoder_input_dict(self):
        return {
                'inputs': tf.placeholder(tf.int32, shape=[self.batch_size, None], name='encoder_inputs'),
               }

    def _decoder_input_dict(self, encoder_output_dict):
        return {
                'inputs': tf.placeholder(tf.int32, shape=[self.batch_size, None], name='decoder_inputs'),
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
            self.targets = tf.placeholder(tf.int32, shape=[self.batch_size, None], name='targets')

            # Outputs (accessed through sess.run)
            self.decoder_final_state = decoder_output_dict['final_state']
            # Loss
            self.logits = decoder_output_dict['logits']
            self.loss = self.compute_loss(self.logits, self.targets)

            # Feedable tensors (feed in through feed_dict)
            self.encoder_init_state = encoder.init_state
            self.decoder_init_state = decoder.init_state

        return encoder_input_dict, encoder_output_dict, decoder_input_dict, decoder_output_dict

    def get_prediction(self, logits, graph=None):
        '''
        Return predicted vocab from output/logits (batch_size, seq_len, vocab_size).
        '''
        preds = np.argmax(logits, axis=2)  # (batch_size, seq_len)
        return preds

    # TODO: put this in evaluator
    def generate(self, sess, batch, encoder_init_state, max_len, graphs=None, utterances=None):
        # Initial feed dict, with encoder inputs, and </t> to start the decoder
        decoder_inputs_last_inds = np.zeros([self.batch_size], dtype=np.int32)
        feed_dict = self.update_feed_dict(encoder_inputs=batch['encoder_inputs'],
                                          decoder_inputs=batch['decoder_inputs'][:, [0]],
                                          encoder_inputs_last_inds=batch['encoder_inputs_last_inds'],
                                          decoder_inputs_last_inds=decoder_inputs_last_inds,
                                          encoder_init_state=encoder_init_state)
        if graphs is not None:
            graph_data = graphs.get_batch_data(batch['encoder_tokens'], None, utterances)
            feed_dict = self.update_feed_dict(feed_dict=feed_dict,
                    encoder_entities=graph_data['encoder_entities'],
                    encoder_input_utterances=graph_data['utterances'])
            self.add_graph_data(feed_dict, graph_data)

        preds = np.zeros([self.batch_size, max_len], dtype=np.int32)
        true_final_state = None
        for i in xrange(max_len):
            if graphs is not None and i == 0:
                # Use the same context through generation
                logits, final_state = sess.run([self.logits, self.decoder_final_state], feed_dict=feed_dict)
            else:
                logits, final_state = sess.run([self.logits, self.decoder_final_state], feed_dict=feed_dict)
            # After step 0, we will use our prediction as input instead of the ground true
            if i == 0:
                true_final_state = final_state
            decoder_inputs = self.get_prediction(logits)
            preds[:, [i]] = decoder_inputs
            feed_dict = self.update_feed_dict(decoder_inputs=decoder_inputs,
                                              decoder_inputs_last_inds=decoder_inputs_last_inds,
                                              decoder_init_state=final_state)

        # Decode over the ground truth
        # -1 because we start from the second input token
        decoder_inputs_last_inds = batch['decoder_inputs_last_inds'] - 1
        # NOTE: we get <0 indices when this is a padded turn, i.e. </t> <pad> <pad> ...
        # At real test time, we want to use batch_size = 1 so this shouldn't happen
        decoder_inputs_last_inds[decoder_inputs_last_inds < 0] = 0
        feed_dict = self.update_feed_dict(decoder_inputs=batch['decoder_inputs'][:, 1:],
                                          decoder_inputs_last_inds=decoder_inputs_last_inds,
                                          decoder_init_state=true_final_state)
        if graphs is not None:
            # Since the graph structure may change during decoding, we cannot continue from
            # the encoder's state. So we just run it all over again.
            # Since the graph has been updated by encoder_tokens, here we only update it using
            # the decoder_tokens. The utterances is the utterances before runing through the
            # encoder, and it is resized according to the new graph structure if necessary.
            new_graph_data = graphs.get_batch_data(None, batch['decoder_tokens'], utterances)
            feed_dict = self.update_feed_dict(encoder_inputs=batch['encoder_inputs'],
                    decoder_inputs=batch['decoder_inputs'],
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
            input_dict['utterances'] = tf.placeholder(tf.float32, shape=[self.batch_size, None, self.graph_embedder.config.utterance_size], name='utterances')
            input_dict['entities'] = tf.placeholder(tf.int32, shape=[self.batch_size, self.graph_embedder.config.entity_cache_size], name='encoder_entities')
        return input_dict

    def _decoder_input_dict(self, encoder_output_dict):
        with tf.name_scope(type(self).__name__+'/decoder_input_dict'):
            input_dict = super(GraphEncoderDecoder, self)._decoder_input_dict(encoder_output_dict)
            # This is used to compute the initial attention
            input_dict['init_output'] = self.encoder._get_final_state(encoder_output_dict['outputs'])
            input_dict['utterances'] = encoder_output_dict['utterances']
            input_dict['entities'] = tf.placeholder(tf.int32, shape=[self.batch_size, self.graph_embedder.config.entity_cache_size], name='decoder_entities')
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
            # Outputs
            self.encoder_output_utterances = encoder_output_dict['utterances']
            self.decoder_output_utterances = decoder_output_dict['utterances']
            # Updated context at the end of the encoder/decoder sequence
            self.encoder_context = encoder_output_dict['context']

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

#class AttnCopyEncoderDecoder(AttnEncoderDecoder):
#    '''
#    Encoder-decoder RNN with attention + copy mechanism over a sequence with conditional write.
#    Attention context is built from knowledge graph (Graph object).
#    Optionally copy from an entity in the knowledge graph.
#    '''
#
#    def __init__(self, vocab, rnn_size, kg, rnn_type='lstm', num_layers=1, scoring='linear', output='project'):
#        super(AttnCopyEncoderDecoder, self).__init__(vocab, rnn_size, kg, rnn_type, num_layers, scoring, output)
#
#    def _build_output(self, h, h_size):
#        token_scores = super(AttnCopyEncoderDecoder, self)._build_output(h, h_size)
#        h, attn_scores = h
#        return tf.concat(1, [token_scores, attn_scores])
#
#    def update_feed_dict(self, feed_dict, inputs, init_state, targets=None, graph=None, iswrite=None):
#        # Don't update targets in parent
#        super(AttnCopyEncoderDecoder, self).update_feed_dict(feed_dict, inputs, init_state, graph=graph, iswrite=iswrite)
#        if targets is not None:
#            feed_dict[self.targets] = graph.copy_target(targets, iswrite, self.vocab)
#
#    def get_prediction(self, outputs, graph):
#        preds = super(AttnCopyEncoderDecoder, self).get_prediction(outputs)
#        preds = graph.copy_output(preds, self.vocab)
#        return preds
