'''
NN models that take a sequence of words and actions.
Encode when action is read and decode when action is write.
'''

import tensorflow as tf
import numpy as np
from model.attention_rnn_cell import AttnRNNCell, add_attention_arguments
from tensorflow.python.util import nest

def add_model_arguments(parser):
    parser.add_argument('--model', default='encdec', help='Model name {encdec}')
    parser.add_argument('--rnn-size', type=int, default=128, help='Dimension of hidden units of RNN')
    parser.add_argument('--rnn-type', default='lstm', help='Type of RNN unit {rnn, gru, lstm}')
    parser.add_argument('--num-layers', default=1, help='Number of RNN layers')
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

    def __init__(self, vocab_size, rnn_size, rnn_type='lstm', num_layers=1, scope=None, para_iter=10):
        # NOTE: only support single-instance training now
        # due to tf.cond(scalar,..)
        self.batch_size = 1
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        self.rnn_size = rnn_size
        self.num_layers = num_layers

        # for scan
        self.parallel_iteration = para_iter

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

    def build_model(self, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            cell = self._build_rnn_cell()

            # Create input variables
            inputs = self._build_rnn_inputs()
            rnn_outputs, states = tf.scan(lambda a, x: cell(x, a[1]), inputs, initializer=(tf.zeros([self.batch_size, cell.output_size]), self.init_state), parallel_iterations=self.parallel_iteration)
            # Get last state
            self.final_state = self._get_final_state(states)

            # Other variables
            self.input_iswrite = tf.placeholder(tf.bool, shape=[self.batch_size, None])
            self.targets = tf.placeholder(tf.int32, shape=[self.batch_size, None])

            # Create output parameters
            w = tf.get_variable('output_w', [cell.output_size, self.vocab_size])
            b = tf.get_variable('output_b', [self.vocab_size])

            # Conditional decoding (only when write is true)
            def cond_output((h, write)):
                '''
                Project RNN state to prediction when write is true
                '''
                def enc():
                    return tf.constant(0, dtype=tf.float32, shape=[self.batch_size, self.vocab_size])
                def dec():
                    return tf.matmul(h, w) + b
                return tf.cond(tf.identity(tf.reshape(write, [])), dec, enc)

            # Used as condition in tf.cond
            iswrite = time_major(self.input_iswrite, 2)

            self.outputs = tf.map_fn(cond_output,
                    (rnn_outputs, iswrite),
                    dtype=tf.float32)

            # Condition loss (loss is 0 when write is false)
            def cond_loss((output, target, write)):
                def loss():
                    return tf.nn.sparse_softmax_cross_entropy_with_logits(output, target)
                def skip():
                    return tf.constant(0, dtype=tf.float32, shape=[self.batch_size])
                return tf.cond(tf.identity(tf.reshape(write, [])), loss, skip)

            # Average loss (per symbol) over the sequence
            # NOTE: should compute average over sequences when batch_size > 1
            self.seq_loss = tf.map_fn(cond_loss,
                    (self.outputs, time_major(self.targets, 2), iswrite),
                    dtype=tf.float32)
            self.loss = tf.reduce_sum(self.seq_loss) / self.batch_size / tf.to_float(tf.shape(self.seq_loss)[0])

    def generate(self, sess, inputs, stop_symbols, max_len=None, init_state=None):
        # Encode inputs
        feed_dict = {}
        if init_state:
            feed_dict[self.init_state] = init_state
        if hasattr(self, 'kg'):
            feed_dict[self.kg.input_data] = self.kg.paths
        if inputs.shape[1] > 1:
            # Read until the second last token, the last one will
            # be used as the first input during decoding
            feed_dict[self.input_data] = inputs[:, :-1]
            [state] = sess.run([self.final_state], feed_dict=feed_dict)
        else:
            state = init_state

        # Decode outputs
        iswrite = np.ones([1, 1]).astype(np.bool_)  # True
        preds = []
        # Last token in the inputs
        input_ = np.expand_dims(inputs[:, -1], 1)
        while True:
            feed_dict = {self.input_data: input_,
                    self.input_iswrite: iswrite}
            if state is not None:
                feed_dict[self.init_state] = state
            if hasattr(self, 'kg'):
                feed_dict[self.kg.input_data] = self.kg.paths
            # output is logits of shape seq_len x batch_size x vocab_size
            # Here both seq_len and batch_size is 1
            state, output = sess.run([self.final_state, self.outputs], feed_dict=feed_dict)
            # pred is of shape (1, 1)
            pred = np.argmax(output, axis=2)
            input_ = pred
            pred = int(pred)
            preds.append(pred)
            if pred in stop_symbols or len(preds) == max_len:
                break
        return preds, state

    def test(self):
        seq_len = 4
        data = np.random.randint(self.vocab_size, size=(self.batch_size, seq_len+1))
        x = data[:,:-1]
        y = data[:,1:]
        iswrite = np.random.randint(2, size=(self.batch_size, seq_len)).astype(np.bool_)

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            feed_dict = {self.input_data: x,
                    self.input_iswrite: iswrite,
                    self.targets: y}
            if hasattr(self, 'kg'):
                feed_dict[self.kg.input_data] = self.kg.paths
            outputs, seq_loss, loss = sess.run([self.outputs, self.seq_loss, model.loss], feed_dict=feed_dict)
            #print 'last_ind:', last_ind
            #print 'states:', states[0].shape, states[1].shape
            print 'is_write:\n', iswrite
            print 'output:\n', outputs.shape, outputs
            print 'seq_loss:\n', seq_loss
            print 'loss:\n', loss
            preds, state = self.generate(sess, x, (5,), 10)
            print 'preds:\n', preds

class AttnEncoderDecoder(EncoderDecoder):
    '''
    Encoder-decoder RNN with attention mechanism over a sequence with conditional write.
    Attention context is built from knowledge graph (Graph object).
    '''

    def __init__(self, vocab_size, rnn_size, kg, rnn_type='lstm', num_layers=1):
        '''
        kg is a Graph object used to compute knowledge graph embeddings.
        '''
        self.context_size = kg.embed_size
        self.kg = kg
        # NOTE: parallel_iteration must be one due to TF issue
        super(AttnEncoderDecoder, self).__init__(vocab_size, rnn_size,  rnn_type, num_layers, para_iter=1)

    def _build_rnn_cell(self):
        cell = AttnRNNCell(self.rnn_size, self.kg, self.rnn_type, num_layers=self.num_layers)

        # Initial state
        self.init_state = cell.zero_state(self.batch_size, tf.float32)

        return cell

# test
if __name__ == '__main__':
    import sys
    # Simple encoder-decoder
    vocab_size = 5
    rnn_size = 10
    model = EncoderDecoder(vocab_size, rnn_size, 'rnn')
    model.test()
    sys.exit()

    # KG + encoder-decoder
    import argparse
    from basic.dataset import add_dataset_arguments, read_dataset
    from basic.schema import Schema
    from basic.scenario_db import ScenarioDB, add_scenario_arguments
    from basic.lexicon import Lexicon
    from basic.util import read_json
    import random
    from model.preprocess import DataGenerator
    from model.kg_embed import CBOWGraph

    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    add_scenario_arguments(parser)
    add_dataset_arguments(parser)
    args = parser.parse_args()
    random.seed(args.random_seed)

    schema = Schema(args.schema_path)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))
    dataset = read_dataset(scenario_db, args)
    lexicon = Lexicon(schema)

    data_generator = DataGenerator(dataset.train_examples, dataset.test_examples, None, lexicon, None)

    gen = data_generator.generator_train('train')
    agent, kb, inputs, targets, iswrite = gen.next()

    context_size = 6
    kg = CBOWGraph(schema, context_size)
    kg.load(kb)
    model = AttnEncoderDecoder(vocab_size, rnn_size, kg, 'lstm')
    model.test()
