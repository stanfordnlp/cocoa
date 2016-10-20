import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops.rnn_cell import _linear as linear
from itertools import izip

def batch_embedding_lookup(embeddings, indices):
    '''
    Look up from a batch of embedding matrices.
    embeddings: (batch_size, num_words, embedding_size)
    indices: (batch_size, num_inds)
    '''
    # Use static shape when possible
    batch_size, num_words, embed_size = embeddings.get_shape().as_list()
    shape = tf.shape(embeddings)
    batch_size = batch_size or shape[0]
    num_words = num_words or shape[1]

    offset = tf.reshape(tf.range(batch_size) * num_words, [batch_size, 1])
    flat_embeddings = tf.reshape(embeddings, [-1, embed_size])
    flat_indices = tf.reshape(indices + offset, [-1])
    embeds = tf.reshape(tf.nn.embedding_lookup(flat_embeddings, flat_indices), [batch_size, -1, embed_size])
    return embeds

def batch_linear(args, output_size, bias):
    '''
    Apply linear map to a batch of matrices.
    args: a 3D Tensor or a list of 3D, batch x n x m, Tensors.
    '''
    if not nest.is_sequence(args):
        args = [args]
    batch_size = args[0].get_shape().as_list()[0] or tf.shape(args[0])[0]
    flat_args = []
    for arg in args:
        m = arg.get_shape().as_list()[2]
        if not m:
            raise ValueError('batch_linear expects shape[2] of arguments: %s' % str(m))
        flat_args.append(tf.reshape(arg, [-1, m]))
    flat_output = linear(flat_args, output_size, bias)
    output = tf.reshape(flat_output, [batch_size, -1, output_size])
    return output

def transpose_first_two_dims(batch_input):
    rank = len(batch_input.get_shape().as_list())
    return tf.transpose(batch_input, perm=[1, 0]+range(2, rank))

