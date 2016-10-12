import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops.rnn_cell import _linear as linear

def batch_embedding_lookup(embeddings, indices):
    '''
    Look up from a batch of embedding matrices.
    embeddings: (batch_size, num_words, embedding_size)
    indices: (batch_size, num_inds)
    '''
    batch_size, num_words, embed_size = embeddings.get_shape().as_list()
    if num_words is not None:
        offset = tf.reshape(tf.range(batch_size) * num_words, [batch_size, 1])
        flat_embeddings = tf.reshape(embeddings, [-1, embed_size])
        flat_indices = tf.reshape(indices + offset, [-1])
        embeds = tf.reshape(tf.nn.embedding_lookup(flat_embeddings, flat_indices), [batch_size, -1, embed_size])
    else:
        print 'Warning (batch_embedding_lookup): number of words in each batch is unknown. Using for loop instead of batching.'
        embeds = []
        for i in xrange(batch_size):
            embeds.append(tf.nn.embedding_lookup(embeddings[i], indices[i]))
        embeds = tf.pack(embeds)
    return embeds

def batch_linear(args, output_size, bias):
    '''
    Apply linear map to a batch of matrices.
    args: a 3D Tensor or a list of 3D, batch x n x m, Tensors.
    '''
    if not nest.is_sequence(args):
        args = [args]
    batch_size = args[0].get_shape().as_list()[0]
    flat_args = []
    for arg in args:
        m = arg.get_shape().as_list()[2]
        if not m:
            raise ValueError('batch_linear expects shape[2] of arguments: %s' % str(m))
        flat_args.append(tf.reshape(arg, [-1, m]))
    flat_output = linear(flat_args, output_size, bias)
    output = tf.reshape(flat_output, [batch_size, -1, output_size])
    return output
