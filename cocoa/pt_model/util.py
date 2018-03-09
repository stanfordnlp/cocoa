from itertools import izip
import numpy as np
import torch
from torch.autograd import Variable

EPS = 1e-12
use_cuda = torch.cuda.is_available()

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

def aeq(*args): # are equal?
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

def use_gpu(opt):
    return (hasattr(opt, 'gpuid') and len(opt.gpuid) > 0) or \
        (hasattr(opt, 'gpu') and opt.gpu > -1)

def safe_div(numerator, denominator):
    return numerator / (denominator + EPS)

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
