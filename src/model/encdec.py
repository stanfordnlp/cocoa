'''
NN models that take a sequence of words and actions.
Encode when action is read and decode when action is write.
'''

import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest
from src.model.rnn_cell import AttnRNNCell, add_attention_arguments, build_rnn_cell
from src.model.graph import Graph, GraphMetadata
from src.model.graph_embedder import GraphEmbedder, GraphEmbedderConfig
from src.model.word_embedder import WordEmbedder
from src.model.util import transpose_first_two_dims, batch_linear, batch_embedding_lookup
from src.model.preprocess import markers

def add_model_arguments(parser):
    parser.add_argument('--model', default='encdec', help='Model name {encdec}')
    parser.add_argument('--rnn-size', type=int, default=20, help='Dimension of hidden units of RNN')
    parser.add_argument('--rnn-type', default='lstm', help='Type of RNN unit {rnn, gru, lstm}')
    parser.add_argument('--num-layers', type=int, default=1, help='Number of RNN layers')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of examples per batch')
    parser.add_argument('--word-embed-size', type=int, default=20, help='Word embedding size')
    parser.add_argument('--encoder-zero-init-state', default=False, action='store_true', help='The encoder state starts from zero for an utterance, instead of continuing from the previous state.')
    parser.add_argument('--gated-copy', default=False, action='store_true', help='Use gating function for copy')
    parser.add_argument('--sup-gate', default=False, action='store_true', help='Supervise copy gate')
    add_attention_arguments(parser)

def build_model(schema, mappings, args):
    tf.reset_default_graph()
    tf.set_random_seed(args.random_seed)

    vocab = mappings['vocab']
    pad = vocab.to_ind(markers.PAD)
    word_embedder = WordEmbedder(vocab.size, args.word_embed_size)
    if args.model == 'encdec':
        encoder = BasicEncoder(args.rnn_size, args.rnn_type, args.num_layers, args.encoder_zero_init_state)
        decoder = BasicDecoder(args.rnn_size, vocab.size, args.rnn_type, args.num_layers)
        model = BasicEncoderDecoder(word_embedder, encoder, decoder, pad)
    elif args.model == 'attn-encdec' or args.model == 'attn-copy-encdec':
        max_degree = args.num_items + len(schema.attributes)
        graph_metadata = GraphMetadata(schema, mappings['entity'], mappings['relation'], args.rnn_size, args.max_num_entities, max_degree=max_degree, entity_hist_len=args.entity_hist_len, entity_cache_size=args.entity_cache_size, num_items=args.num_items)
        graph_embedder_config = GraphEmbedderConfig(args.node_embed_size, args.edge_embed_size, graph_metadata, entity_embed_size=args.entity_embed_size, use_entity_embedding=args.use_entity_embedding, mp_iters=args.mp_iters, decay=args.utterance_decay)
        Graph.metadata = graph_metadata
        graph_embedder = GraphEmbedder(graph_embedder_config)
        encoder = GraphEncoder(args.rnn_size, graph_embedder, rnn_type=args.rnn_type, num_layers=args.num_layers, zero_init_state=args.encoder_zero_init_state)
        if args.model == 'attn-encdec':
            decoder = GraphDecoder(args.rnn_size, vocab.size, graph_embedder, rnn_type=args.rnn_type, num_layers=args.num_layers)
        elif args.model == 'attn-copy-encdec':
            if args.gated_copy:
                decoder = GatedCopyGraphDecoder(args.rnn_size, vocab.size, graph_embedder, rnn_type=args.rnn_type, num_layers=args.num_layers)
                sup_gate = args.sup_gate
            else:
                decoder = CopyGraphDecoder(args.rnn_size, vocab.size, graph_embedder, rnn_type=args.rnn_type, num_layers=args.num_layers)
                sup_gate = False
        model = GraphEncoderDecoder(word_embedder, graph_embedder, encoder, decoder, pad, sup_gate)
    else:
        raise ValueError('Unknown model')
    return model

def get_prediction(logits):
    '''
    Return predicted vocab from output/logits (batch_size, seq_len, vocab_size).
    '''
    preds = np.argmax(logits, axis=2)  # (batch_size, seq_len)
    return preds

def optional_add(feed_dict, key, value):
    if value is not None:
        feed_dict[key] = value

EPS = 1e-12

class BasicEncoder(object):
    '''
    A basic RNN encoder.
    '''
    def __init__(self, rnn_size, rnn_type='lstm', num_layers=1, zero_init_state=False):
        self.rnn_size = rnn_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.zero_init_state = zero_init_state

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
            flat_last_states = []
            for state in flat_states:
                state = transpose_first_two_dims(state)  # (batch_size, time_seq, state_size)
                # NOTE: when state has dim=4, it's the context which does not change in a seq; just take the last one.
                if len(state.get_shape()) == 4:
                    last_state = state[:, -1, :, :]
                else:
                    last_state = tf.squeeze(batch_embedding_lookup(state, tf.reshape(self.last_inds, [-1, 1])), [1])
                flat_last_states.append(last_state)
            last_states = nest.pack_sequence_as(states, flat_last_states)
        return last_states

    def _build_rnn_cell(self):
        return build_rnn_cell(self.rnn_type, self.rnn_size, self.num_layers)

    def _build_init_state(self, cell, input_dict):
        initial_state = input_dict['init_state']
        if initial_state is not None:
            return initial_state
        else:
            return cell.zero_state(self.batch_size, tf.float32)

    def _build_rnn_inputs(self, word_embedder, time_major):
        inputs = word_embedder.embed(self.inputs)
        if not time_major:
            inputs = transpose_first_two_dims(inputs)  # (seq_len, batch_size, input_size)
        return inputs

    def _build_inputs(self, input_dict):
        with tf.name_scope(type(self).__name__+'/inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=[None, None], name='inputs')
            self.last_inds = tf.placeholder(tf.int32, shape=[None], name='last_inds')

    def build_model(self, word_embedder, input_dict, time_major=True, scope=None):
        '''
        inputs: (batch_size, seq_len, input_size)
        '''
        with tf.variable_scope(scope or type(self).__name__):
            self._build_inputs(input_dict)
            self.batch_size = tf.shape(self.inputs)[0]

            cell = self._build_rnn_cell()
            self.init_state = self._build_init_state(cell, input_dict)
            self.output_size = cell.output_size

            inputs = self._build_rnn_inputs(word_embedder, time_major)

            rnn_outputs, states = tf.scan(lambda a, x: cell(x, a[1]), inputs, initializer=(self._build_init_output(cell), self.init_state))

        self.output_dict = self._build_output_dict(rnn_outputs, states)

    def _build_output_dict(self, rnn_outputs, rnn_states):
        final_state = self._get_final_state(rnn_states)
        return {'outputs': rnn_outputs, 'final_state': final_state}

    def get_feed_dict(self, **kwargs):
        feed_dict = kwargs.pop('feed_dict', {})
        feed_dict[self.inputs] = kwargs.pop('inputs')
        feed_dict[self.last_inds] = kwargs.pop('last_inds')
        if not self.zero_init_state:
            optional_add(feed_dict, self.init_state, kwargs.pop('init_state', None))
        return feed_dict

    def run(self, sess, fetches, feed_dict):
        results = sess.run([self.output_dict[x] for x in fetches], feed_dict=feed_dict)
        return {k: results[i] for i, k in enumerate(fetches)}

    def encode(self, sess, **kwargs):
        feed_dict = self.get_feed_dict(**kwargs)
        return self.run(sess, ('final_state',), feed_dict)

class GraphEncoder(BasicEncoder):
    '''
    RNN encoder that update knowledge graph at the end.
    '''
    def __init__(self, rnn_size, graph_embedder, rnn_type='lstm', num_layers=1, zero_init_state=False):
        super(GraphEncoder, self).__init__(rnn_size, rnn_type, num_layers, zero_init_state)
        self.graph_embedder = graph_embedder
        # Id of the utterance matrix to be updated: 0 is encoder utterances, 1 is decoder utterances
        self.utterance_id = 0

    def _build_inputs(self, input_dict):
        super(GraphEncoder, self)._build_inputs(input_dict)
        with tf.name_scope(type(self).__name__+'/inputs'):
            self.utterances = (tf.placeholder(tf.float32, shape=[None, None, self.graph_embedder.config.utterance_size], name='encoder_utterances'),
                    tf.placeholder(tf.float32, shape=[None, None, self.graph_embedder.config.utterance_size], name='decoder_utterances'))
            self.entities = tf.placeholder(tf.int32, shape=[None, self.graph_embedder.config.entity_cache_size], name='entities')

    def build_model(self, word_embedder, input_dict, time_major=True, scope=None):
        super(GraphEncoder, self).build_model(word_embedder, input_dict, time_major=time_major, scope=scope)

        # Use the final encoder state as the utterance embedding
        final_output = self._get_final_state(self.output_dict['outputs'])
        new_utterances = self.graph_embedder.update_utterance(self.entities, final_output, self.utterances, self.utterance_id)
        with tf.variable_scope(tf.get_variable_scope(), reuse=self.graph_embedder.context_initialized):
            context = self.graph_embedder.get_context(new_utterances)

        self.output_dict['utterances'] = new_utterances
        self.output_dict['context'] = context
        self.output_dict['final_output'] = final_output

    def get_feed_dict(self, **kwargs):
        feed_dict = super(GraphEncoder, self).get_feed_dict(**kwargs)
        optional_add(feed_dict, self.utterances, kwargs.pop('utterances', None))
        optional_add(feed_dict, self.entities, kwargs.pop('entities', None))
        return feed_dict

    def encode(self, sess, **kwargs):
        feed_dict = self.get_feed_dict(**kwargs)
        feed_dict = self.graph_embedder.get_feed_dict(feed_dict=feed_dict, **kwargs['graph_data'])
        return self.run(sess, ('final_state', 'final_output', 'utterances', 'context'), feed_dict)

class BasicDecoder(BasicEncoder):
    def __init__(self, rnn_size, num_symbols, rnn_type='lstm', num_layers=1, zero_init_state=False):
        super(BasicDecoder, self).__init__(rnn_size, rnn_type, num_layers, zero_init_state)
        self.num_symbols = num_symbols

    def _build_output(self, output_dict):
        '''
        Take RNN outputs and produce logits over the vocab.
        '''
        outputs = output_dict['outputs']
        outputs = transpose_first_two_dims(outputs)  # (batch_size, seq_len, output_size)
        logits = batch_linear(outputs, self.num_symbols, True)
        return logits

    def build_model(self, word_embedder, input_dict, time_major=True, scope=None):
        super(BasicDecoder, self).build_model(word_embedder, input_dict, time_major=time_major, scope=scope)  # outputs: (seq_len, batch_size, output_size)
        with tf.variable_scope(scope or type(self).__name__):
            logits = self._build_output(self.output_dict)
        self.output_dict['logits'] = logits

    def pred_to_input(self, preds, **kwargs):
        '''
        Convert predictions to input of the next decoding step.
        '''
        textint_map = kwargs.pop('textint_map')
        inputs = textint_map.pred_to_input(preds)
        return inputs

    def decode(self, sess, max_len, batch_size=1, stop_symbol=None, **kwargs):
        if stop_symbol is not None:
            assert batch_size == 1, 'Early stop only works for single instance'
        feed_dict = self.get_feed_dict(**kwargs)
        preds = np.zeros([batch_size, max_len], dtype=np.int32)
        # last_inds=0 because input length is one from here on
        last_inds = np.zeros([batch_size], dtype=np.int32)
        for i in xrange(max_len):
            logits, final_state = sess.run((self.output_dict['logits'], self.output_dict['final_state']), feed_dict=feed_dict)
            step_preds = get_prediction(logits)
            preds[:, [i]] = step_preds
            if step_preds[0][0] == stop_symbol:
                break
            feed_dict = self.get_feed_dict(inputs=self.pred_to_input(step_preds, **kwargs),
                    last_inds=last_inds,
                    init_state=final_state)
        return {'preds': preds, 'final_state': final_state}

class GraphDecoder(GraphEncoder):
    '''
    Decoder with attention mechanism over the graph.
    '''
    def __init__(self, rnn_size, num_symbols, graph_embedder, rnn_type='lstm', num_layers=1, zero_init_state=False, scoring='linear', output='project'):
        super(GraphDecoder, self).__init__(rnn_size, graph_embedder, rnn_type, num_layers, zero_init_state)
        self.num_symbols = num_symbols
        self.utterance_id = 1
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

    def _build_init_state(self, cell, input_dict):
        self.init_output = input_dict['init_output']
        self.init_rnn_state = input_dict['init_state']
        if self.init_rnn_state is not None:
            # NOTE: we assume that the initial state comes from the encoder and is just
            # the rnn state. We need to compute attention and get context for the attention
            # cell's initial state.
            return cell.init_state(self.init_rnn_state, self.init_output, self.context, self.checklists[:, 0, :])
            #return cell.init_state(self.init_rnn_state, self.init_output, self.context)
        else:
            return cell.zero_state(self.batch_size, self.context)

    def compute_init_state(self, sess, init_rnn_state, init_output, context, checklists):
        init_state = sess.run(self.init_state,
                feed_dict={self.init_output: init_output,
                    self.init_rnn_state: init_rnn_state,
                    self.context: context,
                    self.checklists: checklists}
                )
        return init_state

    def _build_inputs(self, input_dict):
        # NOTE: we're calling grandfather's method here because utterances is built differently
        super(GraphEncoder, self)._build_inputs(input_dict)
        with tf.name_scope(type(self).__name__+'/inputs'):
            # Continue from the encoder
            self.utterances = input_dict['utterances']
            self.context = input_dict['context']
            # Inputs
            self.entities = tf.placeholder(tf.int32, shape=[None, self.graph_embedder.config.entity_cache_size], name='entities')
            self.checklists = tf.placeholder(tf.float32, shape=[None, None, None], name='checklists')
            self.copied_nodes = (tf.placeholder(tf.int32, shape=[None, None], name='copied_nodes'), tf.placeholder(tf.bool, shape=[None, None], name='copied_nodes_mask'))

    def _build_rnn_inputs(self, word_embedder, time_major):
        inputs = word_embedder.embed(self.inputs)
        if not time_major:
            inputs = transpose_first_two_dims(inputs)  # (seq_len, batch_size, input_size)
            checklists = transpose_first_two_dims(self.checklists)  # (seq_len, batch_size, num_nodes)
            copied_nodes = (transpose_first_two_dims(self.copied_nodes[0]),
                    transpose_first_two_dims(self.copied_nodes[1]))  # (seq_len, batch_size)
        return (inputs, checklists, copied_nodes)
        #return inputs

    def build_model(self, word_embedder, input_dict, time_major=True, scope=None):
        super(GraphDecoder, self).build_model(word_embedder, input_dict, time_major=time_major, scope=scope)  # outputs: (seq_len, batch_size, output_size)
        with tf.variable_scope(scope or type(self).__name__):
            logits = self._build_output(self.output_dict)
        self.output_dict['logits'] = logits

    def _build_output_dict(self, rnn_outputs, rnn_states):
        final_state = self._get_final_state(rnn_states)
        outputs, attn_scores = rnn_outputs
        return {'outputs': outputs, 'attn_scores': attn_scores, 'final_state': final_state}

    def get_feed_dict(self, **kwargs):
        feed_dict = super(GraphDecoder, self).get_feed_dict(**kwargs)
        feed_dict[self.checklists] = kwargs.pop('checklists')
        feed_dict[self.copied_nodes] = kwargs.pop('copied_nodes')
        return feed_dict

    def pred_to_input(self, preds, **kwargs):
        '''
        Convert predictions to input of the next decoding step.
        '''
        textint_map = kwargs.pop('textint_map')
        inputs = textint_map.pred_to_input(preds)
        return inputs

    def update_checklist(self, pred, cl, graphs, vocab):
        graphs.update_checklist(pred, cl, vocab)

    def get_copied_nodes(self, pred, graphs, vocab):
        return graphs.get_copied_nodes(pred, vocab)

    def decode(self, sess, max_len, batch_size=1, stop_symbol=None, **kwargs):
        if stop_symbol is not None:
            assert batch_size == 1, 'Early stop only works for single instance'
        feed_dict = self.get_feed_dict(**kwargs)
        cl = kwargs['checklists']
        copied_nodes = kwargs['copied_nodes']
        preds = np.zeros([batch_size, max_len], dtype=np.int32)
        # last_inds=0 because input length is one from here on
        last_inds = np.zeros([batch_size], dtype=np.int32)
        attn_scores = []
        graphs = kwargs['graphs']
        vocab = kwargs['vocab']
        for i in xrange(max_len):
            #self._print_cl(cl)
            #self._print_copied_nodes(copied_nodes)
            logits, final_state, final_output, attn_score = sess.run([self.output_dict['logits'], self.output_dict['final_state'], self.output_dict['final_output'], self.output_dict['attn_scores']], feed_dict=feed_dict)
            # attn_score: seq_len x batch_size x num_nodes, seq_len=1, so we take attn_score[0]
            attn_scores.append(attn_score[0])
            step_preds = get_prediction(logits)
            preds[:, [i]] = step_preds
            if step_preds[0][0] == stop_symbol:
                break
            self.update_checklist(step_preds, cl[:, 0, :], graphs, vocab)
            copied_nodes = self.get_copied_nodes(step_preds, graphs, vocab)
            feed_dict = self.get_feed_dict(inputs=self.pred_to_input(step_preds, **kwargs),
                    last_inds=last_inds,
                    init_state=final_state,
                    checklists=cl,
                    copied_nodes=copied_nodes)
        return {'preds': preds, 'final_state': final_state, 'final_output': final_output, 'attn_scores': attn_scores}

    def _print_cl(self, cl):
        print 'checklists:'
        for i in xrange(cl.shape[0]):
            nodes = []
            for j, c in enumerate(cl[i][0]):
                if c != 0:
                    nodes.append(j)
            if len(nodes) > 0:
                print i, nodes

    def _print_copied_nodes(self, cn):
        print 'copied_nodes:'
        cn, mask = cn
        for i, (c, m) in enumerate(zip(cn, mask)):
            if m:
                print i, c

    def update_utterances(self, sess, entities, final_output, utterances, graph_data):
        feed_dict = {self.entities: entities,
                self.output_dict['final_output']: final_output,
                self.utterances: utterances}
        feed_dict = self.graph_embedder.get_feed_dict(feed_dict=feed_dict, **graph_data)
        new_utterances = sess.run(self.output_dict['utterances'], feed_dict=feed_dict)
        return new_utterances

class CopyGraphDecoder(GraphDecoder):
    '''
    Decoder with copy mechanism over the attention context.
    '''
    def _build_output(self, output_dict):
        '''
        Take RNN outputs and produce logits over the vocab and the attentions.
        '''
        logits = super(CopyGraphDecoder, self)._build_output(output_dict)  # (batch_size, seq_len, num_symbols)
        attn_scores = transpose_first_two_dims(output_dict['attn_scores'])  # (batch_size, seq_len, num_nodes)
        return tf.concat(2, [logits, attn_scores])

    def update_checklist(self, pred, cl, graphs, vocab):
        '''
        Update checklist for a single time step.
        '''
        pred = graphs.copy_preds(pred, vocab.size)
        graphs.update_checklist(pred, cl, vocab)

    def get_copied_nodes(self, pred, graphs, vocab):
        '''
        Return copied nodes for a single time step.
        '''
        pred = graphs.copy_preds(pred, vocab.size)
        copied_nodes, mask = graphs.get_zero_copied_nodes(1)
        graphs.update_copied_nodes(pred[:, 0], copied_nodes, mask, vocab)
        return copied_nodes, mask

    def pred_to_input(self, preds, **kwargs):
        graphs = kwargs.pop('graphs')
        vocab = kwargs.pop('vocab')
        textint_map = kwargs.pop('textint_map')
        preds = graphs.copy_preds(preds, vocab.size)
        preds = textint_map.pred_to_input(preds)
        return preds

class GatedCopyGraphDecoder(GraphDecoder):
    '''
    Decoder with copy mechanism over the attention context, where there is an additional gating
    function deciding whether to generate from the vocab or to copy from the graph.
    '''
    def build_model(self, word_embedder, input_dict, time_major=True, scope=None):
        super(GatedCopyGraphDecoder, self).build_model(word_embedder, input_dict, time_major=time_major, scope=scope)  # outputs: (seq_len, batch_size, output_size)
        logits, gate_logits = self.output_dict['logits']
        self.output_dict['logits'] = logits
        self.output_dict['gate_logits'] = gate_logits

    def _build_output(self, output_dict):
        vocab_logits = super(GatedCopyGraphDecoder, self)._build_output(output_dict)  # (batch_size, seq_len, num_symbols)
        attn_scores = transpose_first_two_dims(output_dict['attn_scores'])  # (batch_size, seq_len, num_nodes)
        rnn_outputs = transpose_first_two_dims(output_dict['outputs'])  # (batch_size, seq_len, output_size)
        with tf.variable_scope('Gating'):
            prob_vocab = tf.sigmoid(batch_linear(rnn_outputs, 1, True))  # (batch_size, seq_len, 1)
            prob_copy = 1 - prob_vocab
            log_prob_vocab = tf.log(prob_vocab + EPS)
            log_prob_copy = tf.log(prob_copy + EPS)
        # Reweight the vocab and attn distribution and convert them to logits
        vocab_logits = log_prob_vocab + vocab_logits - tf.reduce_logsumexp(vocab_logits, 2, keep_dims=True)
        attn_logits = log_prob_copy + attn_scores - tf.reduce_logsumexp(attn_scores, 2, keep_dims=True)
        return tf.concat(2, [vocab_logits, attn_logits]), tf.concat(2, [log_prob_vocab, log_prob_copy])

class BasicEncoderDecoder(object):
    '''
    Basic seq2seq model.
    '''
    def __init__(self, word_embedder, encoder, decoder, pad, scope=None):
        self.PAD = pad  # Id of PAD in the vocab
        self.encoder = encoder
        self.decoder = decoder
        self.build_model(word_embedder, encoder, decoder, scope)

    def compute_loss(self, output_dict, targets):
        return self._compute_loss(output_dict['logits'], targets)

    def _compute_loss(self, logits, targets):
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
        token_weights_sum = tf.reduce_sum(tf.reshape(token_weights, [batch_size, -1]), 1) + EPS
        # Average over words in each sequence
        loss = tf.reduce_sum(tf.reshape(loss, [batch_size, -1]), 1) / token_weights_sum

        # Mask padded turns
        seq_weights = tf.cast(tf.not_equal(tf.reshape(targets, [batch_size, -1])[:, 0], tf.constant(self.PAD)), tf.float32)
        seq_loss = loss * seq_weights
        seq_weights_sum = tf.reduce_sum(seq_weights) + EPS
        # Average over sequences in the batch
        loss = tf.reduce_sum(seq_loss, 0) / seq_weights_sum
        return loss, seq_loss

    def _encoder_input_dict(self):
        return {
                'init_state': None,
               }

    def _decoder_input_dict(self, encoder_output_dict):
        return {
                'init_state': encoder_output_dict['final_state'],
               }

    def build_model(self, word_embedder, encoder, decoder, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # Encoding
            with tf.name_scope('Encoder'):
                encoder_input_dict = self._encoder_input_dict()
                encoder.build_model(word_embedder, encoder_input_dict, time_major=False)

            # Decoding
            with tf.name_scope('Decoder'):
                decoder_input_dict = self._decoder_input_dict(encoder.output_dict)
                decoder.build_model(word_embedder, decoder_input_dict, time_major=False)

            self.targets = tf.placeholder(tf.int32, shape=[None, None], name='targets')

            # Loss
            self.loss, self.seq_loss = self.compute_loss(decoder.output_dict, self.targets)

    def get_feed_dict(self, **kwargs):
        feed_dict = kwargs.pop('feed_dict', {})
        feed_dict = self.encoder.get_feed_dict(**kwargs.pop('encoder'))
        feed_dict = self.decoder.get_feed_dict(feed_dict=feed_dict, **kwargs.pop('decoder'))
        optional_add(feed_dict, self.targets, kwargs.pop('targets', None))
        return feed_dict

    def generate(self, sess, batch, encoder_init_state, max_len, copy=False, vocab=None, graphs=None, utterances=None, textint_map=None):
        encoder_inputs = batch['encoder_inputs']
        decoder_inputs = batch['decoder_inputs']
        batch_size = encoder_inputs.shape[0]

        # Encode true prefix
        encoder_args = {'inputs': encoder_inputs,
                'last_inds': batch['encoder_inputs_last_inds'],
                'init_state': encoder_init_state
                }
        if graphs:
            graph_data = graphs.get_batch_data(batch['encoder_tokens'], None, utterances)
            encoder_args['entities'] = graph_data['encoder_entities']
            encoder_args['utterances'] = graph_data['utterances']
            encoder_args['graph_data'] = graph_data
        encoder_output_dict = self.encoder.encode(sess, **encoder_args)

        # Decode max_len steps
        decoder_args = {'inputs': decoder_inputs[:, [0]],
                'last_inds': np.zeros([batch_size], dtype=np.int32),
                'init_state': encoder_output_dict['final_state'],
                'textint_map': textint_map
                }
        if graphs:
            checklists = graphs.get_zero_checklists(1)
            decoder_args['init_state'] = self.decoder.compute_init_state(sess,
                    encoder_output_dict['final_state'],
                    encoder_output_dict['final_output'],
                    encoder_output_dict['context'],
                    checklists)
            decoder_args['checklists'] = checklists
            decoder_args['copied_nodes'] = graphs.get_zero_copied_nodes(1)
            decoder_args['graphs'] = graphs
            decoder_args['vocab'] = vocab
        decoder_output_dict = self.decoder.decode(sess, max_len, batch_size, **decoder_args)

        # Decode true utterances (so that we always condition on true prefix)
        decoder_args['inputs'] = decoder_inputs
        decoder_args['last_inds'] = batch['decoder_inputs_last_inds']
        if graphs is not None:
            # TODO: why do we need to do encoding again
            # Read decoder tokens and update graph
            new_graph_data = graphs.get_batch_data(None, batch['decoder_tokens'], utterances)
            # Add checklists
            checklists = graphs.get_checklists(batch['targets'], vocab)
            decoder_args['checklists'] = checklists
            # Add copied nodes
            copied_nodes = graphs.get_copied_nodes(batch['targets'], vocab)
            decoder_args['copied_nodes'] = copied_nodes
            # Update utterance matrix size and decoder entities given the true decoding sequence
            encoder_args['utterances'] = new_graph_data['utterances']
            decoder_args['entities'] = new_graph_data['decoder_entities']
            # Continue from encoder state, don't need init_state
            decoder_args.pop('init_state')
            kwargs = {'encoder': encoder_args, 'decoder': decoder_args, 'graph_embedder': new_graph_data}
            feed_dict = self.get_feed_dict(**kwargs)
            true_final_state, utterances = sess.run((self.decoder.output_dict['final_state'], self.decoder.output_dict['utterances']), feed_dict=feed_dict)
        else:
            feed_dict = self.decoder.get_feed_dict(**decoder_args)
            true_final_state = sess.run((self.decoder.output_dict['final_state']), feed_dict=feed_dict)

        return decoder_output_dict['preds'], decoder_output_dict['final_state'], true_final_state, utterances, decoder_output_dict['attn_scores']

class GraphEncoderDecoder(BasicEncoderDecoder):
    def __init__(self, word_embedder, graph_embedder, encoder, decoder, pad, sup_gate=None, scope=None):
        self.graph_embedder = graph_embedder
        self.sup_gate = sup_gate
        super(GraphEncoderDecoder, self).__init__(word_embedder, encoder, decoder, pad, scope)

    def _decoder_input_dict(self, encoder_output_dict):
        input_dict = super(GraphEncoderDecoder, self)._decoder_input_dict(encoder_output_dict)
        # This is used to compute the initial attention
        input_dict['init_output'] = self.encoder._get_final_state(encoder_output_dict['outputs'])
        input_dict['utterances'] = encoder_output_dict['utterances']
        input_dict['context'] = encoder_output_dict['context']
        return input_dict

    def get_feed_dict(self, **kwargs):
        feed_dict = super(GraphEncoderDecoder, self).get_feed_dict(**kwargs)
        feed_dict = self.graph_embedder.get_feed_dict(feed_dict=feed_dict, **kwargs['graph_embedder'])
        return feed_dict

    def compute_loss(self, output_dict, targets):
        loss, seq_loss = super(GraphEncoderDecoder, self).compute_loss(output_dict, targets)
        if self.sup_gate:
            vocab_size = self.decoder.num_symbols
            # 0: vocab 1: copy
            targets = tf.cast(tf.greater_equal(targets, vocab_size), tf.int32)
            gate_loss, gate_seq_loss = self._compute_loss(output_dict['gate_logits'], targets)
            loss += gate_loss
            seq_loss += gate_seq_loss
        return loss, seq_loss

