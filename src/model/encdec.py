'''
NN models that take a sequence of words and actions.
Encode when action is read and decode when action is write.
'''

import tensorflow as tf
import numpy as np
from attention_rnn_cell import AttnRNNCell, add_attention_arguments
from graph import Graph
from graph_embed import GraphEmbed
from vocab import is_entity
from tensorflow.python.util import nest
from itertools import izip
import sys

def add_model_arguments(parser):
    parser.add_argument('--model', default='encdec', help='Model name {encdec}')
    parser.add_argument('--rnn-size', type=int, default=128, help='Dimension of hidden units of RNN')
    parser.add_argument('--rnn-type', default='lstm', help='Type of RNN unit {rnn, gru, lstm}')
    parser.add_argument('--num-layers', type=int, default=1, help='Number of RNN layers')
    add_attention_arguments(parser)

def time_major(batch_input, rank):
    '''
    Input: tensor of shape [batch_size, seq_len, ..]
    Output: tensor of shape [seq_len, batch_size, ..]
    Time-major shape is used for map_fn and dynamic_rnn.
    '''
    return tf.transpose(batch_input, perm=[1, 0]+range(2, rank))

class EncoderDecoder(object):
    '''
    Basic encoder-decoder RNN over a sequence with conditional write.
    '''
    recurrent_cell = {'rnn': tf.nn.rnn_cell.BasicRNNCell,
                      'gru': tf.nn.rnn_cell.GRUCell,
                      'lstm': tf.nn.rnn_cell.LSTMCell,
                     }

    def __init__(self, vocab_size, rnn_size, rnn_type='lstm', num_layers=1, scope=None):
        # NOTE: only support single-instance training now
        # due to tf.cond(scalar,..)
        self.batch_size = 1
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        self.rnn_size = rnn_size
        self.num_layers = num_layers

        self.build_model(scope)

    def _build_rnn_cell(self):
        '''
        Create the internal multi-layer recurrent cell and specify the initial state.
        '''
        cell = None
        if self.rnn_type == 'lstm':
            cell = EncoderDecoder.recurrent_cell[self.rnn_type](self.rnn_size, state_is_tuple=True)
        else:
            cell = EncoderDecoder.recurrent_cell[self.rnn_type](self.rnn_size)
        if self.num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers)

        # Initial state
        self.init_state = cell.zero_state(self.batch_size, tf.float32)

        return cell

    def _build_rnn_inputs(self):
        '''
        Create input data placeholder(s), inputs to rnn and
        needed variables (e.g., for embedding).
        '''
        self.input_data = tf.placeholder(tf.int32, shape=[self.batch_size, None])
        embedding = tf.get_variable('embedding', [self.vocab_size, self.rnn_size])
        rnn_inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        return time_major(rnn_inputs, 3)

    def _get_final_state(self, states):
        '''
        Return the final state from tf.scan outputs.
        '''
        flat_states = nest.flatten(states)
        last_ind = tf.shape(flat_states[0])[0] - 1
        flat_last_states = [tf.nn.embedding_lookup(state, last_ind) for state in flat_states]
        last_states = nest.pack_sequence_as(states, flat_last_states)
        return last_states

    def _build_init_output(self, cell):
        '''
        Initializer for scan. Should have the same shape as the RNN output.
        '''
        return tf.zeros([self.batch_size, cell.output_size])

    def build_model(self, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            cell = self._build_rnn_cell()

            # Create input variables
            inputs = self._build_rnn_inputs()
            self.inputs = inputs
            rnn_outputs, states = tf.scan(lambda a, x: cell(x, a[1]), inputs, initializer=(self._build_init_output(cell), self.init_state))
            self.states = states
            # Get last state
            self.final_state = self._get_final_state(states)

            # Other variables
            self.input_iswrite = tf.placeholder(tf.bool, shape=[self.batch_size, None])
            self.targets = tf.placeholder(tf.int32, shape=[self.batch_size, None])

            # Conditional decoding (only when write is true)
            def cond_output((h, write)):
                '''
                Project RNN state to prediction when write is true
                '''
                dec_output = self._build_output(h, cell.output_size)
                def enc():
                    return tf.zeros_like(dec_output)
                def dec():
                    return dec_output
                return tf.cond(tf.reshape(write, []), dec, enc)

            # Used as condition in tf.cond
            iswrite = time_major(self.input_iswrite, 2)

            outputs = tf.map_fn(cond_output,
                    (rnn_outputs, iswrite),
                    dtype=tf.float32)
            # Change output shape to batch_size x time_step
            self.outputs = tf.transpose(outputs, [1, 0, 2])

            # Condition loss (loss is 0 when write is false)
            def cond_loss((output, target, write)):
                def loss():
                    return tf.nn.sparse_softmax_cross_entropy_with_logits(output, target)
                def skip():
                    return tf.constant(0, dtype=tf.float32, shape=[self.batch_size])
                return tf.cond(tf.reshape(write, []), loss, skip)

            # Average loss (per symbol) over the sequence
            # NOTE: should compute average over sequences when batch_size > 1
            self.seq_loss = tf.map_fn(cond_loss,
                    (outputs, time_major(self.targets, 2), iswrite),
                    dtype=tf.float32)
            self.loss = tf.reduce_sum(self.seq_loss) / self.batch_size / tf.to_float(tf.shape(self.seq_loss)[0])

    def _build_output(self, h, h_size):
        '''
        Take RNN outputs (h) and output logits over the vocab.
        '''
        # Create output parameters
        w = tf.get_variable('output_w', [h_size, self.vocab_size])
        b = tf.get_variable('output_b', [self.vocab_size])
        return tf.matmul(h, w) + b

    def update_feed_dict(self, feed_dict, inputs, init_state, targets=None, graph=None, iswrite=None):
        feed_dict[self.input_data] = inputs
        if init_state is not None:
            feed_dict[self.init_state] = init_state
        if iswrite is not None:
            feed_dict[self.input_iswrite] = iswrite
        if targets is not None:
            feed_dict[self.targets] = targets

    def get_prediction(self, outputs, graph=None):
        '''
        Return predicted vocab from output/logits (batch_size x time_step x vocab_size).
        '''
        preds = np.argmax(outputs, axis=2)  # batch_size x time_step
        return preds

    def generate(self, sess, inputs, graph, stop_symbols, max_len=None, init_state=None):
        # Encode inputs
        feed_dict = {}
        # Read until the second last token, the last one will be used as the first input during decoding
        self.update_feed_dict(feed_dict, inputs[:, :-1], init_state, graph=graph)
        if inputs.shape[1] > 1:
            [state] = sess.run([self.final_state], feed_dict=feed_dict)
        else:
            state = init_state

        # Decode outputs
        iswrite = np.ones([1, 1]).astype(np.bool_)  # True
        # Last token in the inputs; use -1: to keep dimension the same
        input_ = inputs[:, -1:]
        # kb has been updatd, no need to update again
        self.update_feed_dict(feed_dict, input_, state, graph=graph, iswrite=iswrite)
        preds = []

        while True:
            # output is logits of shape seq_len x batch_size x vocab_size
            # Here both seq_len and batch_size is 1
            state, output = sess.run([self.final_state, self.outputs], feed_dict=feed_dict)
            # next input is the prediction: shape (1, 1)
            input_ = self.get_prediction(output, graph)
            pred = int(input_)
            assert pred < self.vocab_size
            preds.append(pred)
            # NOTE: Check if len(preds) > 1 to avoid generating stop_symbols as the first token
            if (len(preds) > 1 and pred in stop_symbols) or len(preds) == max_len:
                break

            self.update_feed_dict(feed_dict, input_, state, graph=graph)

        return preds, state

    def test(self, kb=None, vocab=None):
        seq_len = 4
        np.random.seed(0)
        data = np.random.randint(self.vocab_size, size=(self.batch_size, seq_len+1))
        x = data[:,:-1]
        y = data[:,1:]
        iswrite = np.random.randint(2, size=(self.batch_size, seq_len)).astype(np.bool_)
        if kb and vocab:
            graph = Graph(kb)
        else:
            graph = None

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            feed_dict = {}
            self.update_feed_dict(feed_dict, x, None, graph=graph, targets=y, iswrite=iswrite)
            outputs, seq_loss, loss = sess.run([self.outputs, self.seq_loss, model.loss], feed_dict=feed_dict)
            #print 'last_ind:', last_ind
            #print 'states:', states[0].shape, states[1].shape
            print 'is_write:\n', iswrite
            print 'output:\n'
            assert outputs.shape[1] == seq_len
            if graph:
                assert outputs.shape[2] == vocab.size + graph.nodes.size
            print 'seq_loss:\n', seq_loss
            print 'loss:\n', loss
            preds, state = self.generate(sess, x, graph, (5,), max_len=10)
            print 'preds:\n', preds

class AttnEncoderDecoder(EncoderDecoder):
    '''
    Encoder-decoder RNN with attention mechanism over a sequence with conditional write.
    Attention context is built from knowledge graph (Graph object).
    '''

    def __init__(self, vocab, rnn_size, kg, rnn_type='lstm', num_layers=1, scoring='linear', output='project'):
        '''
        kg is a GraphEmbed object used to compute knowledge graph embeddings.
        '''
        self.kg = kg
        self.context_size = self.kg.node_embed_size
        assert Graph.setup
        self.scoring_method = scoring
        self.output_method = output
        self.vocab = vocab
        super(AttnEncoderDecoder, self).__init__(vocab.size, rnn_size,  rnn_type, num_layers)

    def _build_rnn_inputs(self):
        '''
        Input includes tokens and entities.
        Each token has a correponding entity list, i.e., the previously mentioned n entities.
        Each entity is mapped to an interger according to lexicon.entity_to_ind.
        '''
        input_tokens = super(AttnEncoderDecoder, self)._build_rnn_inputs()

        input_entity_shape = self.kg.input_entity_shape
        self.input_entities = tf.placeholder(tf.int32, shape=input_entity_shape)
        input_entities = time_major(self.input_entities, len(input_entity_shape))

        self.input_updates = tf.placeholder(tf.bool, shape=self.input_data.get_shape())
        input_updates = time_major(self.input_updates, 2)

        return (input_tokens, input_entities, input_updates)

    def _build_rnn_cell(self):
        cell = AttnRNNCell(self.rnn_size, self.kg, self.rnn_type, num_layers=self.num_layers, scoring=self.scoring_method, output=self.output_method)

        # Initial state
        self.init_state = cell.zero_state(self.batch_size, tf.float32)

        return cell

    def _build_output(self, h, h_size):
        '''
        Take RNN outputs (h) and output logits over the vocab.
        '''
        h, attn_scores = h
        return super(AttnEncoderDecoder, self)._build_output(h, h_size)

    def _build_init_output(self, cell):
        '''
        Output includes both RNN output and attention scores.
        '''
        output = super(AttnEncoderDecoder, self)._build_init_output(cell)
        return (output, tf.zeros_like(tf.reshape(self.kg.node_ids, [1, -1]), dtype=tf.float32))

    def update_feed_dict(self, feed_dict, inputs, init_state, targets=None, graph=None, iswrite=None):
        super(AttnEncoderDecoder, self).update_feed_dict(feed_dict, inputs, init_state, targets=targets, iswrite=iswrite)
        # TODO: input should be tokens instead of ints
        tokens = map(self.vocab.to_word, list(inputs[0]))  # NOTE: assumes batch_size=1
        graph.read_utterance(tokens)
        feed_dict[self.input_entities] = self.kg.reshape_input_entity(graph.get_entity_list(len(tokens)))
        feed_dict[self.kg.input_data] = graph.get_input_data()
        updates = np.zeros(inputs.shape, dtype=np.bool)
        feed_dict[self.input_updates] = updates

class AttnCopyEncoderDecoder(AttnEncoderDecoder):
    '''
    Encoder-decoder RNN with attention + copy mechanism over a sequence with conditional write.
    Attention context is built from knowledge graph (Graph object).
    Optionally copy from an entity in the knowledge graph.
    '''

    def __init__(self, vocab, rnn_size, kg, rnn_type='lstm', num_layers=1, scoring='linear', output='project'):
        super(AttnCopyEncoderDecoder, self).__init__(vocab, rnn_size, kg, rnn_type, num_layers, scoring, output)

    def _build_output(self, h, h_size):
        token_scores = super(AttnCopyEncoderDecoder, self)._build_output(h, h_size)
        h, attn_scores = h
        return tf.concat(1, [token_scores, attn_scores])

    def update_feed_dict(self, feed_dict, inputs, init_state, targets=None, graph=None, iswrite=None):
        # Don't update targets in parent
        super(AttnCopyEncoderDecoder, self).update_feed_dict(feed_dict, inputs, init_state, graph=graph, iswrite=iswrite)
        if targets is not None:
            feed_dict[self.targets] = graph.copy_target(targets, iswrite, self.vocab)

    def get_prediction(self, outputs, graph):
        preds = super(AttnCopyEncoderDecoder, self).get_prediction(outputs)
        preds = graph.copy_output(preds, self.vocab)
        return preds

# test
if __name__ == '__main__':
    # Simple encoder-decoder
    vocab_size = 5
    rnn_size = 10
    model = EncoderDecoder(vocab_size, rnn_size, 'rnn')
    model.test()

    from basic.schema import Schema
    from model.preprocess import build_schema_mappings
    from basic.kb import KB
    from model.vocab import Vocabulary

    schema = Schema('data/friends-schema.json')
    entity_map, relation_map = build_schema_mappings(schema)
    max_degree = 3

    items = [{'Name': 'Alice', 'Company': 'Microsoft', 'Hobby': 'hiking'},\
             {'Name': 'Bob', 'Company': 'Apple', 'Hobby': 'hiking'}]
    kb = KB.from_dict(schema, items)

    Graph.static_init(schema, entity_map, relation_map, max_degree)
    graph = Graph(kb)

    vocab = Vocabulary()
    vocab.add_words([('alice', 'person'), ('bob', 'person'), ('microsoft', 'company'), ('apple', 'company'), ('hiking', 'hobby')])

    tf.reset_default_graph()
    node_embed_size = 10
    edge_embed_size = 9
    with tf.Graph().as_default():
        tf.set_random_seed(0)
        kg = GraphEmbed(100, Graph.relation_map.size, node_embed_size, edge_embed_size, rnn_size, Graph.feat_size)
        model = AttnCopyEncoderDecoder(vocab, rnn_size, kg)
        model.test(kb, vocab)
