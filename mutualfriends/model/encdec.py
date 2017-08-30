import tensorflow as tf
import numpy as np
from cocoa.model.util import transpose_first_two_dims, batch_linear, batch_embedding_lookup, EPS
from cocoa.model.encdec import BasicEncoder, BasicDecoder, Sampler, optional_add
# Task-specific modules
from rnn_cell import AttnRNNCell, PreselectAttnRNNCell
from preprocess import markers

class GraphEncoder(BasicEncoder):
    '''
    RNN encoder that update knowledge graph at the end.
    '''
    def __init__(self, rnn_size, graph_embedder, rnn_type='lstm', num_layers=1, dropout=0, node_embed_in_rnn_inputs=False, update_graph=True, separate_utterance_embedding=False):
        super(GraphEncoder, self).__init__(rnn_size, rnn_type, num_layers, dropout)
        self.graph_embedder = graph_embedder
        self.context_size = self.graph_embedder.config.context_size
        # Id of the utterance matrix to be updated: 0 is encoder utterances, 1 is decoder utterances
        self.utterance_id = 0
        self.node_embed_in_rnn_inputs = node_embed_in_rnn_inputs
        self.update_graph = update_graph
        self.separate_utterance_embedding = separate_utterance_embedding

    def _build_graph_variables(self, input_dict):
        if 'utterances' in input_dict:
            self.utterances = input_dict['utterances']
        else:
            self.utterances = (tf.placeholder(tf.float32, shape=[None, None, self.graph_embedder.config.utterance_size], name='encoder_utterances'),
                    tf.placeholder(tf.float32, shape=[None, None, self.graph_embedder.config.utterance_size], name='decoder_utterances'))

        if 'context' in input_dict:
            self.context = input_dict['context']
        else:
            with tf.variable_scope(tf.get_variable_scope(), reuse=self.graph_embedder.context_initialized):
                self.context = self.graph_embedder.get_context(self.utterances)
        self.num_nodes = tf.to_int32(tf.shape(self.context[0])[1])

    def _build_inputs(self, input_dict):
        super(GraphEncoder, self)._build_inputs(input_dict)
        with tf.name_scope('Inputs'):
            # Entities whose embedding are to be updated
            self.update_entities = tf.placeholder(tf.int32, shape=[None, None], name='update_entities')
            # Entities in the current utterance. Non-entity words are -1.
            self.entities = tf.placeholder(tf.int32, shape=[None, None], name='entities')

    def _get_node_embedding(self, context, node_ids):
        '''
        Lookup embeddings of nodes from context.
        node_ids: (batch_size, seq_len)
        context: (batch_size, num_nodes, context_size)
        Return node_embeds (batch_size, seq_len, context_size)
        '''
        node_embeddings = batch_embedding_lookup(context, node_ids, zero_ind=-1)  # (batch_size, seq_len, context_size)
        return node_embeddings

    def get_rnn_inputs_args(self):
        '''
        Return inputs used to build_rnn_inputs for encoding.
        '''
        args = super(GraphEncoder, self).get_rnn_inputs_args()
        args['entities'] = self.entities
        args['context'] = self.context
        return args

    def _build_rnn_inputs(self, time_major, **kwargs):
        '''
        Concatenate word embedding with entity/node embedding.
        '''
        word_embedder = self.word_embedder
        inputs = kwargs.get('inputs', self.inputs)
        entities = kwargs.get('entities', self.entities)
        context = kwargs.get('context', self.context)

        word_embeddings = word_embedder.embed(inputs, zero_pad=True)
        if self.node_embed_in_rnn_inputs:
            # stop_gradien: tLook up node embeddings but don't back propogate (would be recursive)
            entity_embeddings = tf.stop_gradient(self._get_node_embedding(context[0], entities))
            inputs = tf.concat(2, [word_embeddings, entity_embeddings])
        else:
            inputs = word_embeddings
        if not time_major:
            inputs = transpose_first_two_dims(inputs)  # (seq_len, batch_size, input_size)
        return inputs

    def _embed_utterance(self):
        return self.encode(time_major=False, **{'init_state': None})['final_output']

    def build_model(self, word_embedder, input_dict, tf_variables, time_major=True, scope=None):
        # Variable space is GraphEncoderDecoder
        self._build_graph_variables(input_dict)

        # Variable space is type(self)
        super(GraphEncoder, self).build_model(word_embedder, input_dict, tf_variables, time_major=time_major, scope=scope)

        # Variable scope is GraphEncoderDecoder
        # Use the final encoder state as the utterance embedding
        final_output = self._get_final_state(self.output_dict['outputs'])
        if not self.separate_utterance_embedding:
            self.utterance_embedding = final_output
        else:
            # TODO: better ways to share weights
            with tf.variable_scope('UtteranceEmbed', reuse=('UtteranceEmbed' in tf_variables)):
                self.utterance_embedding = self._embed_utterance()
                tf_variables.add('UtteranceEmbed')
        new_utterances = self.graph_embedder.update_utterance(self.update_entities, self.utterance_embedding, self.utterances, self.utterance_id)
        if not self.update_graph:
            new_utterances = self.utterances
        with tf.variable_scope(tf.get_variable_scope(), reuse=self.graph_embedder.context_initialized):
            context = self.graph_embedder.get_context(new_utterances)

        self.output_dict['utterances'] = new_utterances
        self.output_dict['context'] = context
        self.output_dict['final_output'] = final_output

    def get_feed_dict(self, **kwargs):
        feed_dict = super(GraphEncoder, self).get_feed_dict(**kwargs)
        feed_dict[self.entities] = kwargs.pop('entities')
        optional_add(feed_dict, self.utterances, kwargs.pop('utterances', None))
        optional_add(feed_dict, self.update_entities, kwargs.pop('update_entities', None))
        return feed_dict

    def run_encode(self, sess, **kwargs):
        feed_dict = self.get_feed_dict(**kwargs)
        feed_dict = self.graph_embedder.get_feed_dict(feed_dict=feed_dict, **kwargs['graph_data'])
        return self.run(sess, ('final_state', 'final_output', 'utterances', 'context'), feed_dict)

class GraphDecoder(GraphEncoder):
    '''
    Decoder with attention mechanism over the graph.
    '''
    def __init__(self, rnn_size, num_symbols, graph_embedder, rnn_type='lstm', num_layers=1, dropout=0, scoring='linear', output='project', checklist=True, sampler=Sampler(0), node_embed_in_rnn_inputs=False, update_graph=True, separate_utterance_embedding=False, encoder=None):
        super(GraphDecoder, self).__init__(rnn_size, graph_embedder, rnn_type, num_layers, dropout, node_embed_in_rnn_inputs, update_graph, separate_utterance_embedding)
        if self.separate_utterance_embedding:
            assert encoder is not None
            self.encoder = encoder
        self.sampler = sampler
        self.num_symbols = num_symbols
        self.utterance_id = 1
        self.scorer = scoring
        self.output_combiner = output
        self.checklist = checklist

    def get_encoder_state(self, state):
        '''
        Given the hidden state to the encoder to continue from there.
        '''
        # NOTE: state = (rnn_state, attn, context)
        return state[0]

    def _embed_utterance(self):
        kwargs = self.get_rnn_inputs_args()
        kwargs['init_state'] = None
        return self.encoder.encode(time_major=False, **kwargs)['final_output']

    def compute_loss(self, targets, pad, select):
        loss, seq_loss, total_loss = BasicDecoder._compute_loss(pad)
        # -1 is selection loss
        return loss, seq_loss, total_loss, tf.constant(-1.)

    def _build_rnn_cell(self):
        return AttnRNNCell(self.rnn_size, self.context_size, self.rnn_type, self.keep_prob, self.scorer, self.output_combiner, self.num_layers, self.checklist)

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
        #logits = BasicDecoder.penalize_repetition(logits)
        return logits

    def _build_init_state(self, cell, input_dict):
        self.init_output = input_dict['init_output']
        self.init_rnn_state = input_dict['init_state']
        if self.init_rnn_state is not None:
            # NOTE: we assume that the initial state comes from the encoder and is just
            # the rnn state. We need to compute attention and get context for the attention
            # cell's initial state.
            return cell.init_state(self.init_rnn_state, self.init_output, self.context, tf.cast(self.init_checklists[:, 0, :], tf.float32))
        else:
            return cell.zero_state(self.batch_size, self.context)

    # TODO: hacky interface
    def compute_init_state(self, sess, init_rnn_state, init_output, context, init_checklists):
        init_state = sess.run(self.init_state,
                feed_dict={self.init_output: init_output,
                    self.init_rnn_state: init_rnn_state,
                    self.context: context,
                    self.init_checklists: init_checklists,}
                )
        return init_state

    def _build_inputs(self, input_dict):
        super(GraphDecoder, self)._build_inputs(input_dict)
        with tf.name_scope('Inputs'):
            self.init_checklists = tf.placeholder(tf.int32, shape=[None, None, None], name='init_checklists')
            self.targets = tf.placeholder(tf.int32, shape=[None, None], name='targets')

    def _build_rnn_inputs(self, time_major):
        inputs = super(GraphDecoder, self)._build_rnn_inputs(time_major)

        checklists = tf.cumsum(tf.one_hot(self.entities, self.num_nodes, on_value=1, off_value=0), axis=1) + self.init_checklists
        # cumsum can cause >1 indicator
        checklists = tf.cast(tf.greater(checklists, 0), tf.float32)
        self.output_dict['checklists'] = checklists

        checklists = transpose_first_two_dims(checklists)  # (seq_len, batch_size, num_nodes)
        return inputs, checklists

    def build_model(self, word_embedder, input_dict, tf_variables, time_major=True, scope=None):
        super(GraphDecoder, self).build_model(word_embedder, input_dict, tf_variables, time_major=time_major, scope=scope)  # outputs: (seq_len, batch_size, output_size)
        with tf.variable_scope(scope or type(self).__name__):
            logits = self._build_output(self.output_dict)
        self.output_dict['logits'] = logits
        self.output_dict['probs'] = tf.nn.softmax(logits)

    def _build_output_dict(self, rnn_outputs, rnn_states):
        final_state = self._get_final_state(rnn_states)
        outputs, attn_scores = rnn_outputs
        self.output_dict.update({'outputs': outputs, 'attn_scores': attn_scores, 'final_state': final_state})

    def get_feed_dict(self, **kwargs):
        feed_dict = super(GraphDecoder, self).get_feed_dict(**kwargs)
        feed_dict[self.init_checklists] = kwargs.pop('init_checklists')
        optional_add(feed_dict, self.targets, kwargs.pop('targets', None))
        return feed_dict

    def pred_to_input(self, preds, **kwargs):
        '''
        Convert predictions to input of the next decoding step.
        '''
        textint_map = kwargs.pop('textint_map')
        inputs = textint_map.pred_to_input(preds)
        return inputs

    def pred_to_entity(self, pred, graphs, vocab):
        return graphs.pred_to_entity(pred, vocab.size)

    def run_decode(self, sess, max_len, batch_size=1, stop_symbol=None, **kwargs):
        if stop_symbol is not None:
            assert batch_size == 1, 'Early stop only works for single instance'
        feed_dict = self.get_feed_dict(**kwargs)
        cl = kwargs['init_checklists']
        preds = np.zeros([batch_size, max_len], dtype=np.int32)
        generated_word_types = None
        # last_inds=0 because input length is one from here on
        last_inds = np.zeros([batch_size], dtype=np.int32)
        attn_scores = []
        probs = []
        graphs = kwargs['graphs']
        vocab = kwargs['vocab']
        select = vocab.to_ind(markers.SELECT)

        for i in xrange(max_len):
            output_nodes = [self.output_dict['logits'], self.output_dict['final_state'], self.output_dict['final_output'], self.output_dict['attn_scores'], self.output_dict['probs'], self.output_dict['checklists']]
            if 'selection_scores' in self.output_dict:
                output_nodes.append(self.output_dict['selection_scores'])

            if 'selection_scores' in self.output_dict:
                logits, final_state, final_output, attn_score, prob, cl, selection_scores = sess.run(output_nodes, feed_dict=feed_dict)
            else:
                logits, final_state, final_output, attn_score, prob, cl = sess.run(output_nodes, feed_dict=feed_dict)

            # attn_score: seq_len x batch_size x num_nodes, seq_len=1, so we take attn_score[0]
            attn_scores.append(attn_score[0])
            # probs: batch_size x seq_len x num_symbols
            probs.append(prob[:, 0, :])
            step_preds = self.sampler.sample(logits, prev_words=None)

            if generated_word_types is None:
                generated_word_types = np.zeros([batch_size, logits.shape[2]])
            generated_word_types[np.arange(batch_size), step_preds[:, 0]] = 1

            preds[:, [i]] = step_preds
            if step_preds[0][0] == stop_symbol:
                break
            entities = self.pred_to_entity(step_preds, graphs, vocab)

            feed_dict = self.get_feed_dict(inputs=self.pred_to_input(step_preds, **kwargs),
                    last_inds=last_inds,
                    init_state=final_state,
                    init_checklists=cl,
                    entities=entities,
                    )
        # NOTE: the final_output may not be at the stop symbol when the function is running
        # in batch mode -- it will be the state at max_len. This is fine since during test
        # we either run with batch_size=1 (real-time chat) or use the ground truth to update
        # the state (see generate()).
        output_dict = {'preds': preds, 'final_state': final_state, 'final_output': final_output, 'attn_scores': attn_scores, 'probs': probs, 'checklists': cl}
        if 'selection_scores' in self.output_dict:
            output_dict['selection_scores'] = selection_scores
        return output_dict

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

    def update_context(self, sess, entities, utterance_embedding, utterances, graph_data):
        feed_dict = {self.update_entities: entities,
                self.utterance_embedding: utterance_embedding,
                self.utterances: utterances}
        feed_dict = self.graph_embedder.get_feed_dict(feed_dict=feed_dict, **graph_data)
        new_utterances, new_context = sess.run([self.output_dict['utterances'], self.output_dict['context']], feed_dict=feed_dict)
        return new_utterances, new_context

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

    def pred_to_entity(self, pred, graphs, vocab):
        '''
        Return copied nodes for a single time step.
        '''
        offset = vocab.size
        pred = graphs.copy_preds(pred, offset)
        node_ids = graphs._pred_to_node_id(pred, offset)
        return node_ids

    def pred_to_input(self, preds, **kwargs):
        graphs = kwargs.pop('graphs')
        vocab = kwargs.pop('vocab')
        textint_map = kwargs.pop('textint_map')
        preds = graphs.copy_preds(preds, vocab.size)
        preds = textint_map.pred_to_input(preds)
        return preds

class PreselectCopyGraphDecoder(CopyGraphDecoder):
    '''
    Decoder that pre-selects a set of entities before generation.
    '''
    def _build_rnn_cell(self):
        return PreselectAttnRNNCell(self.rnn_size, self.context_size, self.rnn_type, self.keep_prob, self.scorer, self.output_combiner, self.num_layers, self.checklist)

    def _get_all_entities(self, entities):
        '''
        entities: (batch_size, seq_len) node_id at each step in the sequence
        Return indicator vector (batch_size, num_nodes) of all entities in the sequence
        '''
        all_entities = tf.cumsum(tf.one_hot(entities, self.num_nodes, on_value=1, off_value=0), axis=1)
        return tf.greater(all_entities, 0)

    def _build_output_dict(self, rnn_outputs, rnn_states):
        final_state = self._get_final_state(rnn_states)
        selection_scores = final_state[-1]
        outputs, attn_scores = rnn_outputs
        self.output_dict.update({'outputs': outputs, 'attn_scores': attn_scores, 'final_state': final_state, 'selection_scores': selection_scores})

    def compute_loss(self, targets, pad, select):
        loss, seq_loss, total_loss, _ = super(PreselectCopyGraphDecoder, self).compute_loss(targets, pad, select)

        entity_targets = self.output_dict['checklists'][:, -1, :]
        entity_logits = self.output_dict['selection_scores']
        mask = self.context[1]
        entity_loss = tf.where(mask, tf.nn.sigmoid_cross_entropy_with_logits(entity_logits, entity_targets), tf.zeros_like(entity_logits))
        #weights = tf.where(tf.equal(entity_targets, 1),
        #        tf.ones_like(entity_targets) * 1.,
        #        tf.ones_like(entity_targets) * 1.)
        #entity_loss = entity_loss * weights
        entity_loss = tf.reduce_sum(entity_loss) / tf.to_float(self.batch_size) / tf.to_float(self.num_nodes)
        loss += entity_loss

        return loss, seq_loss, total_loss, entity_loss

class GatedCopyGraphDecoder(GraphDecoder):
    '''
    Decoder with copy mechanism over the attention context, where there is an additional gating
    function deciding whether to generate from the vocab or to copy from the graph.
    '''
    def build_model(self, encoder_word_embedder, decoder_word_embedder, input_dict, time_major=True, scope=None):
        super(GatedCopyGraphDecoder, self).build_model(encoder_word_embedder, decoder_word_embedder, input_dict, time_major=time_major, scope=scope)  # outputs: (seq_len, batch_size, output_size)
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

    def compute_loss(self, targets, pad, select):
        loss, seq_loss, total_loss, select_loss = super(GatedCopyGraphDecoder, self).compute_loss(targets, pad, select)

        vocab_size = self.num_symbols
        # 0: vocab 1: copy
        targets = tf.cast(tf.greater_equal(targets, vocab_size), tf.int32)
        gate_loss, gate_seq_loss, gate_total_loss  = self._compute_loss(output_dict['gate_logits'], targets)
        loss += gate_loss
        seq_loss += gate_seq_loss
        total_loss += gate_total_loss

        return loss, seq_loss, total_loss, select_loss

class BasicEncoderDecoder(object):
    '''
    Basic seq2seq model.
    '''
    def __init__(self, encoder_word_embedder, decoder_word_embedder, encoder, decoder, pad, select, re_encode=False, scope=None):
        self.PAD = pad  # Id of PAD in the vocab
        self.SELECT = select
        self.encoder = encoder
        self.decoder = decoder
        self.re_encode = re_encode
        self.tf_variables = set()
        self.build_model(encoder_word_embedder, decoder_word_embedder, encoder, decoder, scope)

    def compute_loss(self, output_dict, targets):
        return self.decoder.compute_loss(self.PAD)

    def _encoder_input_dict(self):
        return {
                'init_state': None,
               }

    def _decoder_input_dict(self, encoder_output_dict):
        return {
                'init_state': encoder_output_dict['final_state'],
               }

    def build_model(self, encoder_word_embedder, decoder_word_embedder, encoder, decoder, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # Encoding
            with tf.name_scope('Encoder'):
                encoder_input_dict = self._encoder_input_dict()
                encoder.build_model(encoder_word_embedder, encoder_input_dict, self.tf_variables, time_major=False)

            # Decoding
            with tf.name_scope('Decoder'):
                decoder_input_dict = self._decoder_input_dict(encoder.output_dict)
                decoder.build_model(decoder_word_embedder, decoder_input_dict, self.tf_variables, time_major=False)

            # Re-encode decoded sequence
            # TODO: re-encode is not implemeted in neural_sessions yet
            # TODO: hierarchical
            if self.re_encode:
                input_args = decoder.get_rnn_inputs_args()
                input_args['init_state'] = encoder.output_dict['final_state']
                reencode_output_dict = encoder.encode(time_major=False, **input_args)
                self.final_state = reencode_output_dict['final_state']
            else:
                self.final_state = decoder.get_encoder_state(decoder.output_dict['final_state'])

            #self.targets = tf.placeholder(tf.int32, shape=[None, None], name='targets')
            # TODO: fix this hack
            self.targets = self.decoder.targets

            # Loss
            # TODO: make loss return a dict to accomadate different loss terms
            losses = self.compute_loss(decoder.output_dict, self.targets)
            self.loss, self.seq_loss, self.total_loss = losses[:3]
            try:
                self.select_loss = losses[3]
            except IndexError:
                self.select_loss = -1.

    def get_feed_dict(self, **kwargs):
        feed_dict = kwargs.pop('feed_dict', {})
        feed_dict = self.encoder.get_feed_dict(**kwargs.pop('encoder'))
        feed_dict = self.decoder.get_feed_dict(feed_dict=feed_dict, **kwargs.pop('decoder'))
        #optional_add(feed_dict, self.targets, kwargs.pop('targets', None))
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
            graph_data = graphs.get_batch_data(batch['encoder_tokens'], None, batch['encoder_entities'], None, utterances, vocab)
            encoder_args['update_entities'] = graph_data['encoder_entities']
            encoder_args['entities'] = graph_data['encoder_nodes']
            encoder_args['utterances'] = graph_data['utterances']
            encoder_args['graph_data'] = graph_data
        encoder_output_dict = self.encoder.run_encode(sess, **encoder_args)

        # Decode max_len steps
        decoder_args = {'inputs': decoder_inputs[:, [0]],
                'last_inds': np.zeros([batch_size], dtype=np.int32),
                'init_state': encoder_output_dict['final_state'],
                'textint_map': textint_map
                }
        if graphs:
            init_checklists = graphs.get_zero_checklists(1)
            entities = graphs.get_zero_entities(1)
            decoder_args['init_state'] = self.decoder.compute_init_state(sess,
                    encoder_output_dict['final_state'],
                    encoder_output_dict['final_output'],
                    encoder_output_dict['context'],
                    init_checklists,
                    )
            decoder_args['init_checklists'] = init_checklists
            decoder_args['entities'] = entities
            decoder_args['graphs'] = graphs
            decoder_args['vocab'] = vocab
        decoder_output_dict = self.decoder.run_decode(sess, max_len, batch_size, **decoder_args)

        # Decode true utterances (so that we always condition on true prefix)
        decoder_args['inputs'] = decoder_inputs
        decoder_args['last_inds'] = batch['decoder_inputs_last_inds']
        if graphs is not None:
            # TODO: why do we need to do encoding again
            # Read decoder tokens and update graph
            new_graph_data = graphs.get_batch_data(None, batch['decoder_tokens'], batch['encoder_entities'], batch['decoder_entities'], utterances, vocab)
            decoder_args['encoder_entities'] = new_graph_data['encoder_nodes']
            # Add checklists
            decoder_args['init_checklists'] = graphs.get_zero_checklists(1)
            # Add copied nodes
            decoder_args['entities'] = new_graph_data['decoder_nodes']
            # Update utterance matrix size and decoder entities given the true decoding sequence
            encoder_args['utterances'] = new_graph_data['utterances']
            decoder_args['update_entities'] = new_graph_data['decoder_entities']
            # Continue from encoder state, don't need init_state
            decoder_args.pop('init_state')
            kwargs = {'encoder': encoder_args, 'decoder': decoder_args, 'graph_embedder': new_graph_data}
            feed_dict = self.get_feed_dict(**kwargs)
            true_final_state, utterances, true_checklists = sess.run((self.final_state, self.decoder.output_dict['utterances'], self.decoder.output_dict['checklists']), feed_dict=feed_dict)

            result = {'preds': decoder_output_dict['preds'],
                      'final_state': decoder_output_dict['final_state'],
                      'true_final_state': true_final_state,
                      'utterances': utterances,
                      'attn_scores': decoder_output_dict['attn_scores'],
                      'probs': decoder_output_dict['probs'],
                      }
            if 'selection_scores' in decoder_output_dict:
                result['selection_scores'] = decoder_output_dict['selection_scores']
                result['true_checklists'] = true_checklists
            return result
        else:
            feed_dict = self.decoder.get_feed_dict(**decoder_args)
            # TODO: this is needed by re-encode
            #feed_dict[self.encoder.keep_prob] = 1. - self.encoder.dropout
            true_final_state = sess.run((self.final_state), feed_dict=feed_dict)
            return {'preds': decoder_output_dict['preds'],
                    'final_state': decoder_output_dict['final_state'],
                    'true_final_state': true_final_state,
                    }

class GraphEncoderDecoder(BasicEncoderDecoder):
    def __init__(self, encoder_word_embedder, decoder_word_embedder, graph_embedder, encoder, decoder, pad, select, re_encode=False, sup_gate=None, scope=None):
        self.graph_embedder = graph_embedder
        self.sup_gate = sup_gate
        self.preselect = True if isinstance(decoder, PreselectCopyGraphDecoder) else False
        super(GraphEncoderDecoder, self).__init__(encoder_word_embedder, decoder_word_embedder, encoder, decoder, pad, select, re_encode, scope)

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

