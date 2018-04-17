import time
import torch
import pdb
import numpy as np
import torch.nn as nn
import math
import sys

from onmt.Loss import LossComputeBase
from symbols import markers

class SimpleLossCompute(LossComputeBase):
    """
    Simpler Loss Computation class - does not perform Truncated BPTT,
        removes label_smoothing, confidence-scores and sharding
    """
    def __init__(self, generator, tgt_vocab):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.padding_idx = tgt_vocab.word_to_ind[markers.PAD]
        weight = torch.ones(tgt_vocab.size)
        weight[self.padding_idx] = 0
        self.criterion = nn.NLLLoss(weight, size_average=False)

    def compute_loss(self, target, output):
        scores = self.generator(self._bottle(output))
        gtruth = target.contiguous().view(-1)
        loss = self.criterion(scores, gtruth)
        loss_data = loss.data.clone()
        stats = self._stats(loss_data, scores.data, target.view(-1).data)
        return loss, stats
