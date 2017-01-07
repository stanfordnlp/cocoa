import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops.rnn_cell import _linear as linear
from itertools import izip

EPS = 1e-12

def embedding_lookup(embeddings, indices, zero_ind=None):
    '''
    Same as tf.nn.embedding_lookup except that it returns a zero vector if the
    lookup index is zero_ind (default -1).
    '''
    if zero_ind is None:
        return tf.nn.embedding_lookup(embeddings, indices)
    else:
        mask = tf.equal(indices, zero_ind)
        # Set zero_ind to 0 as it may be out of range of embeddings shape
        indices = tf.where(mask, tf.zeros_like(indices), indices)
        result = tf.nn.embedding_lookup(embeddings, indices)
        result = tf.where(mask, tf.zeros_like(result), result)
        return result

def batch_embedding_lookup(embeddings, indices, zero_ind=None):
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
    embed_size = embed_size or shape[2]

    offset = tf.reshape(tf.range(batch_size) * num_words, [batch_size, 1])
    flat_embeddings = tf.reshape(embeddings, [-1, embed_size])
    flat_indices = tf.reshape(indices + offset, [-1])
    embeds = tf.reshape(embedding_lookup(flat_embeddings, flat_indices, zero_ind), [batch_size, -1, embed_size])
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

