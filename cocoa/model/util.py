import numpy as np

EPS = 1e-12

def safe_div(numerator, denominator):
    return numerator / (denominator + EPS)

def entropy(p, normalized=True):
    p = np.array(p, dtype=np.float32)
    if not normalized:
        p /= np.sum(p)
    ent = -1. * np.sum(p * np.log(p))
    return ent
