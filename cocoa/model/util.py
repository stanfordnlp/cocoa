from itertools import izip
import numpy as np
from tensorflow.python.util import nest
from tensorflow.contrib.rnn.python.ops.rnn_cell import _linear
import tensorflow as tf

EPS = 1e-12

linear = _linear

def safe_div(numerator, denominator):
    return numerator / (denominator + EPS)

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
    rank = batch_input.shape.ndims
    return tf.transpose(batch_input, perm=[1, 0]+range(2, rank))

def _tile_single_tensor(t, multiplier):
    '''
    Tile a single tensor
    (batch, ...) -> (batch * multiplier, ...)
    '''
    shape_t = tf.shape(t)
    tiling = [1] * (t.shape.ndims + 1)
    tiling[1] = multiplier
    tiled = tf.tile(tf.expand_dims(t, 1), tiling)
    tiled = tf.reshape(tiled, tf.concat([[shape_t[0] * multiplier], shape_t[1:]], axis=0))
    # Presearve static shapes
    tiled_static_batch_size = (t.shape[0].value * multiplier if t.shape[0].value is not None else None)
    tiled.set_shape(
        tf.TensorShape(
            [tiled_static_batch_size]).concatenate(t.shape[1:]))
    return tiled

def tile_tensor(t, multiplier):
    '''
    Tile a possibly nested tensor where each tensor has first dimension batch_size.
    '''
    return nest.map_structure(lambda t_: _tile_single_tensor(t_, multiplier), t)

def resettable_metric(metric, scope_name, **metric_args):
    '''
    Originally from https://github.com/tensorflow/tensorflow/issues/4814#issuecomment-314801758
    '''
    with tf.variable_scope(scope_name) as scope:
        metric_op, update_op = metric(**metric_args)
        v = tf.contrib.framework.get_variables(\
                    scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(v)
    return metric_op, update_op, reset_op

def entropy(p, normalized=True):
    p = np.array(p, dtype=np.float32)
    if not normalized:
        p /= np.sum(p)
    ent = -1. * np.sum(p * np.log(p))
    return ent
