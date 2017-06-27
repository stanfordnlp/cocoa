from itertools import izip
import random
from src.lib.bleu import compute_bleu
import numpy as np

class BaseRanker(object):
    def __init__(self):
        self.name = 'ranker'
        self.perplexity = False

    def select(self, batch):
        raise NotImplementedError

class RandomRanker(BaseRanker):
    def select(self, batch):
        responses = [random.choice(c) for candidates in batch['candidates']]
        return responses

class CheatRanker(BaseRanker):
    def select(self, batch):
        candidates = batch['candidates']
        targets = batch['decoder_tokens']
        responses = []
        for c, target in izip(candidates, targets):
            if not len(target) > 0:
                response = []
            else:
                scores = [compute_bleu(r, target) for r in c]
                response = c[np.argmax(scores)]
            responses.append(response)
        return responses
