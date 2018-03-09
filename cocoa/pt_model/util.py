from itertools import izip
import numpy as np
import torch
from torch.autograd import Variable


EPS = 1e-12
use_cuda = torch.cuda.is_available()

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

def basic_variable(data, dtype="long"):
    if dtype == "long":
        tensor = torch.LongTensor(data)
    elif dtype == "float":
        tensor = torch.FloatTensor(data)
    return Variable(tensor)

def smart_variable(data, dtype="tensor"):
    if dtype == "list":
        result = basic_variable(data)
    elif dtype == "tensor":
        result = Variable(data)
    elif dtype == "var":
        result = data

    return result.cuda() if use_cuda else result

def transpose_first_two_dims(batch_input):
    rank = len(batch_input.size)
    return torch.transpose(batch_input, perm=[1, 0]+range(2, rank))


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
