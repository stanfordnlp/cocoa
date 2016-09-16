'''
NN models that take a sequence of words and actions.
Encode when action is read and decode when action is write.
'''

import tensorflow as tf
import numpy as np
from model.attention_rnn_cell import AttnRNNCell, add_attention_arguments
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
                return tf.cond(tf.identity(tf.reshape(write, [])), dec, enc)

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
                return tf.cond(tf.identity(tf.reshape(write, [])), loss, skip)

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

    def update_feed_dict(self, feed_dict, inputs, init_state, targets=None, kb=None, entities=None, iswrite=None, preds=None):
        feed_dict[self.input_data] = inputs
        if init_state is not None:
            feed_dict[self.init_state] = init_state
        if iswrite is not None:
            feed_dict[self.input_iswrite] = iswrite
        if targets is not None:
            feed_dict[self.targets] = targets

    def get_prediction(self, outputs):
        '''
        Return predicted vocab from output/logits (batch_size x time_step x vocab_size).
        '''
        preds = np.argmax(outputs, axis=2)  # batch_size x time_step
        return preds

    def generate(self, sess, kb, inputs, entities, stop_symbols, max_len=None, init_state=None):
        # Encode inputs
        feed_dict = {}
        entities = entities[:, :-1] if entities is not None else None
        self.update_feed_dict(feed_dict, inputs[:, :-1], init_state, kb=kb, entities=entities)
        if inputs.shape[1] > 1:
            # Read until the second last token, the last one will
            # be used as the first input during decoding
            [state] = sess.run([self.final_state], feed_dict=feed_dict)
        else:
            state = init_state

        # Decode outputs
        iswrite = np.ones([1, 1]).astype(np.bool_)  # True
        # Last token in the inputs; use -1: to keep dimension the same
        input_ = inputs[:, -1:]
        entity = entities[:, -1:] if entities is not None else None
        # kb has been updatd, no need to update again
        self.update_feed_dict(feed_dict, input_, state, kb=None, entities=entity, iswrite=iswrite)
        preds = []

        while True:
            # output is logits of shape seq_len x batch_size x vocab_size
            # Here both seq_len and batch_size is 1
            state, output = sess.run([self.final_state, self.outputs], feed_dict=feed_dict)
            # next input is the prediction: shape (1, 1)
            input_ = self.get_prediction(output)
            pred = int(input_)
            assert pred < self.vocab_size
            preds.append(pred)
            if pred in stop_symbols or len(preds) == max_len:
                break

            self.update_feed_dict(feed_dict, input_, state, kb=None, entities=None, preds=preds)

        return preds, state

    def test(self, kb=None, lexicon=None):
        seq_len = 4
        np.random.seed(0)
        data = np.random.randint(self.vocab_size, size=(self.batch_size, seq_len+1))
        x = data[:,:-1]
        y = data[:,1:]
        iswrite = np.random.randint(2, size=(self.batch_size, seq_len)).astype(np.bool_)
        entities = []
        for i in xrange(seq_len):
            entities.append([-1, 1])
        if not self.kg.train_utterance:
            entities = np.asarray(entities, dtype=np.int32).reshape(1, -1, 2)
        else:
            entities = map(self.kg.convert_entity, entities)
            entities = np.asarray(entities, dtype=np.int32).reshape(1, -1, self.kg.entity_cache_size, self.kg.utterance_size, 2)

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            feed_dict = {self.input_data: x,
                    self.input_iswrite: iswrite,
                    self.targets: y}
            if hasattr(self, 'kg'):
                assert kb
                feed_dict[self.kg.input_data] = self.kg.load(kb)
                feed_dict[self.input_entities] = entities
            outputs, seq_loss, loss = sess.run([self.outputs, self.seq_loss, model.loss], feed_dict=feed_dict)
            #print 'last_ind:', last_ind
            #print 'states:', states[0].shape, states[1].shape
            print 'is_write:\n', iswrite
            print 'output:\n', outputs.shape, outputs
            print 'seq_loss:\n', seq_loss
            print 'loss:\n', loss
            preds, state = self.generate(sess, kb, x, entities, (5,), lexicon, 10)
            print 'preds:\n', preds

class AttnEncoderDecoder(EncoderDecoder):
    '''
    Encoder-decoder RNN with attention mechanism over a sequence with conditional write.
    Attention context is built from knowledge graph (Graph object).
    '''

    def __init__(self, vocab, rnn_size, kg, rnn_type='lstm', num_layers=1, scoring='linear', output='project'):
        '''
        kg is a Graph object used to compute knowledge graph embeddings.
        entity_cache_size: number of entities to keep in the update buffer
        '''
        self.kg = kg
        self.context_size = self.kg.context_size
        self.entity_cache_size = kg.entity_cache_size
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
        if self.kg.train_utterance:
            self.input_entities = tf.placeholder(tf.int32, shape=[self.batch_size, None, self.entity_cache_size, self.kg.utterance_size, 2])
            input_entities = time_major(self.input_entities, 5)
        else:
            self.input_entities = tf.placeholder(tf.int32, shape=[self.batch_size, None, self.entity_cache_size])
            input_entities = time_major(self.input_entities, 3)
        return (input_tokens, input_entities)

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
        return (output, tf.zeros_like(self.kg.entity_size))

    def update_feed_dict(self, feed_dict, inputs, init_state, targets=None, kb=None, entities=None, iswrite=None, preds=None):
        super(AttnEncoderDecoder, self).update_feed_dict(feed_dict, inputs, init_state, targets=targets, iswrite=iswrite)
        if kb is not None:
            feed_dict[self.kg.input_data] = self.kg.load(kb)
        if entities is not None:
            feed_dict[self.input_entities] = entities
        else:
            # Compute entity from preds
            assert preds
            tokens = map(self.vocab.to_word, preds)
            entity = self.kg.get_entity_list(tokens, (len(tokens)-1,))
            if not self.kg.train_utterance:
                entity = np.asarray(entity, dtype=np.int32).reshape([1, -1, self.entity_cache_size])
            else:
                entity = np.asarray(entity, dtype=np.int32).reshape([1, -1, self.entity_cache_size, self.kg.utterance_size, 2])
            feed_dict[self.input_entities] = entity

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

    def update_feed_dict(self, feed_dict, inputs, init_state, targets=None, kb=None, entities=None, iswrite=None, preds=None):
        # Don't update targets in parent
        super(AttnCopyEncoderDecoder, self).update_feed_dict(feed_dict, inputs, init_state, kb=kb, entities=entities, iswrite=iswrite, preds=preds)
        if targets is not None:
            feed_dict[self.targets] = self.copy_target(targets, iswrite)

    def get_prediction(self, outputs):
        preds = super(AttnCopyEncoderDecoder, self).get_prediction(outputs)
        preds = self.copy_output(preds)
        return preds

    def copy_output(self, outputs):
        vocab = self.vocab
        for output in outputs:
            for i, pred in enumerate(output):
                if pred >= vocab.size:
                    entity = self.kg.ind_to_label[self.kg.local_to_global_entity[pred - vocab.size]]
                    output[i] = vocab.to_ind(entity)
        return outputs

    def copy_target(self, targets, iswrite):
        vocab = self.vocab
        # Don't change the original targets, will be used later
        new_targets = np.copy(targets)
        for target, write in izip(new_targets, iswrite):
            for i, (t, w) in enumerate(izip(target, write)):
                # NOTE: only replace entities in our kb
                if w:
                    token = vocab.to_word(t)
                    # TODO: use named tuple to represent entities
                    if not isinstance(token, basestring):
                        try:
                            target[i] = vocab.size + self.kg.global_to_local_entity[self.kg.label_to_ind[token]]
                        except KeyError:
                            print token
                            print self.kg.label_to_ind
                            print self.kg.global_to_local_entity
                            sys.exit()
        return new_targets

# test
if __name__ == '__main__':
    # Simple encoder-decoder
    vocab_size = 5
    rnn_size = 10
    #model = EncoderDecoder(vocab_size, rnn_size, 'rnn')
    #model.test()

    # KG + encoder-decoder
    import argparse
    from basic.dataset import add_dataset_arguments, read_dataset
    from basic.schema import Schema
    from basic.scenario_db import ScenarioDB, add_scenario_arguments
    from basic.lexicon import Lexicon
    from basic.util import read_json
    from model.preprocess import DataGenerator
    from model.kg_embed import CBOWGraph

    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    add_scenario_arguments(parser)
    add_dataset_arguments(parser)
    args = parser.parse_args()
    np.random.seed(args.random_seed)

    schema = Schema(args.schema_path)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))
    dataset = read_dataset(scenario_db, args)
    lexicon = Lexicon(schema)

    data_generator = DataGenerator(dataset.train_examples, dataset.test_examples, None, lexicon)
    vocab = data_generator.vocab

    gen = data_generator.generator_train('train')

    tf.reset_default_graph()
    context_size = 6
    with tf.Graph().as_default():
        tf.set_random_seed(args.random_seed)
        kg = CBOWGraph(schema, lexicon, context_size, rnn_size, train_utterance=False)
        data_generator.set_kg(kg)
        agent, kb, inputs, entities, targets, iswrite = gen.next()
        model = AttnCopyEncoderDecoder(vocab, rnn_size, kg)
        # test copy_output and copy target
        kg.load(kb)
        n = 2
        print 'to be copied:', kg.ind_to_label[kg.nodes[n]], lexicon.id_to_entity[kg.nodes[n]]
        t = np.array([[vocab.to_ind(kg.ind_to_label[kg.nodes[n]])]])
        t = model.copy_target(t, [[True]])
        print 'new target:', t
        out = model.copy_output(t)
        print 'copy output:', vocab.to_word(int(out))
        model.test(kb, data_generator.lexicon)
