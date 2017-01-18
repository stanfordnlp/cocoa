import random
import numpy as np
import math

def normalize_weights(weights):
    '''
    [3, 2] => [0.6, 0.4]
    '''
    if len(weights) == 0:
        return []
    s = sum(weights)
    if s == 0:
        print 'WARNING: zero normalization'
        return weights
    return [1.0 * weight / s for weight in weights]

def exp_normalize_weights(weights):
    m = max(weights)
    weights = [math.exp(w - m) for w in weights]  # Ensure no underflow
    return normalize_weights(weights)

def normalize_candidates(candidates):
    '''
    [('a', 2), ('b', 8)] => [('a', 0.2), ('b', 0.8)]
    '''
    s = sum([weight for token, weight in candidates])
    return [(k, weight / s) for k, weight in candidates]

#def sample_candidates(candidates):
#    '''
#    [('a', 2), ('b', 8)] => 'a' or 'b'
#    '''
#    weights = [weight for token, weight in candidates]
#    sums = numpy.array(weights).cumsum()
#    i = sums.searchsorted(random.random() * sums[-1])
#    return candidates[i]

def sorted_candidates(candidates):
    '''
    [('a', 2), ('b', 8)] => [('b', 8), ('a', 2)]
    '''
    return sorted(candidates, key=lambda (token, weight) : weight, reverse=True)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def sample_candidates(candidates, n=1):
    n = min(n, len(candidates))
    weights = softmax([weight for value, weight in candidates])
    values = [value for value, weight in candidates]
    samples = np.random.choice(range(len(values)), n, replace=False, p=weights)
    return [values[i] for i in samples]
