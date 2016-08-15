'''
NN models that take a sequence of words and actions.
Encode when action is read and decode when action is write.
'''

import tensorflow as tf
import numpy as np

def add_model_arguments(parser):
    parser.add_argument('--model', default='encdec', help='Model name {encdec}')
    parser.add_argument('--rnn-size', type=int, default=128, help='Dimension of hidden units of RNN')
    parser.add_argument('--rnn-type', default='lstm', help='Type of RNN unit {rnn, gru, lstm}')
    parser.add_argument('--num-layers', default=1, help='Number of RNN layers')

recurrent_cell = {'rnn': tf.nn.rnn_cell.BasicRNNCell,
                  'gru': tf.nn.rnn_cell.GRUCell,
                  'lstm': tf.nn.rnn_cell.LSTMCell,
                 }

class EncoderDecoder(object):
    def __init__(self, vocab_size, rnn_size, rnn_type='lstm', num_layers=1):
        # NOTE: only support single-instance training now
        # due to tf.cond(scalar,..)
        self.batch_size = 1
        self.rnn_type = rnn_type

        with tf.variable_scope('encdec'):
            # Create the internal multi-layer recurrent cell
            single_cell = recurrent_cell[rnn_type](rnn_size, state_is_tuple=True) if rnn_type == 'lstm' \
                else recurrent_cell[rnn_type](rnn_size)
            self.cell = single_cell if num_layers == 1 \
                else tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
            self.init_state = self.cell.zero_state(self.batch_size, tf.float32)

            # Create input variables
            self.input_data = tf.placeholder(tf.int32, shape=[self.batch_size, None])
            self.input_iswrite = tf.placeholder(tf.bool, shape=[self.batch_size, None])
            self.targets = tf.placeholder(tf.int32, shape=[self.batch_size, None])

            # Create network parameters
            self.w = tf.get_variable('output_w', [rnn_size, vocab_size])
            self.b = tf.get_variable('output_b', [vocab_size])
            embedding = tf.get_variable('embedding', [vocab_size, rnn_size])

            # Encode with embedding
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)
            rnn_outputs, self.final_state = tf.nn.dynamic_rnn(self.cell, inputs, initial_state=self.init_state)

            # Conditional decoding (only when write is true)
            def cond_output((h, write)):
                '''
                Project RNN state to prediction when write is true
                '''
                def enc():
                    return tf.constant(0, dtype=tf.float32, shape=[self.batch_size, vocab_size])
                def dec():
                    return tf.matmul(h, self.w) + self.b
                return tf.cond(tf.identity(tf.reshape(write, [])), dec, enc)

            def time_major(batch_input, rank):
                '''
                Input: tensor of shape [batch_size, seq_len, ..]
                Output: tensor of shape [seq_len, batch_size, ..]
                Time-major shape is used for map_fn.
                '''
                if rank == 2:
                    return tf.transpose(batch_input)
                elif rank == 3:
                    return tf.transpose(batch_input, perm=[1, 0, 2])
                else:
                    return ValueError('Input must have rank 2 or 3.')

            # Used as condition in tf.cond
            iswrite = time_major(self.input_iswrite, 2)

            self.outputs = tf.map_fn(cond_output,
                    (time_major(rnn_outputs, 3), iswrite),
                    dtype=tf.float32)

            # Condition loss (loss is 0 when write is false)
            def cond_loss((output, target, write)):
                def loss():
                    return tf.nn.sparse_softmax_cross_entropy_with_logits(output, target)
                def skip():
                    return tf.constant(0, dtype=tf.float32, shape=[self.batch_size])
                return tf.cond(tf.identity(tf.reshape(write, [])), loss, skip)

            # Average loss (per symbol) over the sequence
            # NOTE: compute average over sequences when batch_size > 1
            self.seq_loss = tf.map_fn(cond_loss,
                    (self.outputs, time_major(self.targets, 2), iswrite),
                    dtype=tf.float32)
            self.loss = tf.reduce_sum(self.seq_loss) / self.batch_size / tf.to_float(tf.shape(self.seq_loss)[0])

    def generate(self, sess, inputs, stop_symbols, max_len=None, init_state=None):
        # Encode inputs
        feed_dict = {}
        if init_state:
            feed_dict[self.init_state] = init_state
        if inputs.shape[1] > 1:
            # Read until the second last token, the last one will
            # be used as the first input during decoding
            feed_dict[self.input_data] = inputs[:, :-1].reshape(1, -1)
            [state] = sess.run([self.final_state], feed_dict=feed_dict)
        else:
            state = init_state

        # Decode outputs
        iswrite = np.ones([1, 1]).astype(np.bool_)  # True
        preds = []
        # Last token in the inputs
        input_ = inputs[:, -1].reshape(-1, 1)
        while True:
            feed_dict = {self.input_data: input_,
                    self.input_iswrite: iswrite}
            if state is not None:
                feed_dict[self.init_state] = state
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


# test
if __name__ == '__main__':
    vocab_size = 5
    rnn_size = 10
    batch_size = 1
    seq_len = 4
    model = EncoderDecoder(vocab_size, rnn_size, 'rnn')
    data = np.random.randint(vocab_size, size=(batch_size, seq_len+1))
    x = data[:,:-1]
    y = data[:,1:]
    iswrite = np.random.randint(2, size=(batch_size, seq_len)).astype(np.bool_)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        feed_dict = {model.input_data: x,
                model.input_iswrite: iswrite,
                model.targets: y}
        outputs, seq_loss, loss = sess.run([model.outputs, model.seq_loss, model.loss], feed_dict=feed_dict)
        print 'is_write:\n', iswrite
        print 'output:\n', outputs.shape, outputs
        print 'seq_loss:\n', seq_loss
        print 'loss:\n', loss
        preds, state = model.generate(sess, x, (5,), 10)
        print 'preds:\n', preds

