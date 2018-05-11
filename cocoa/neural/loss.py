import time
import torch
import torch.nn as nn

from onmt.Loss import LossComputeBase
from onmt.Utils import aeq

from symbols import markers
#from utterance import UtteranceBuilder

class SimpleLossCompute(LossComputeBase):
    """
    Simpler Loss Computation class - does not perform Truncated BPTT,
        removes label_smoothing, confidence-scores and sharding
    """
    def __init__(self, generator, tgt_vocab):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.padding_idx = tgt_vocab.to_ind(markers.PAD)
        weight = torch.ones(tgt_vocab.size)
        weight[self.padding_idx] = 0
        self.criterion = nn.NLLLoss(weight, size_average=False)

    def compute_loss(self, target, output):
        # generator: RNN outputs to vocab_size scores/logprobs
        # output: (seq_len, batch_size, rnn_size)
        scores = self.generator(self._bottle(output))
        gtruth = target.contiguous().view(-1)
        loss = self.criterion(scores, gtruth)
        loss_data = loss.data.clone()
        stats = self._stats(loss_data, scores.data, target.view(-1).data)
        return loss, stats

class ReinforceLossCompute(SimpleLossCompute):
    """Compute loss/reward for REINFORCE.
    """
    def __init__(self, generator, tgt_vocab):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.padding_idx = tgt_vocab.to_ind(markers.PAD)
        weight = torch.ones(tgt_vocab.size)
        weight[self.padding_idx] = 0
        self.criterion = nn.NLLLoss(weight, size_average=False, reduce=False)
        #self.builder = UtteranceBuilder(tgt_vocab)

    def compute_loss(self, target, output):
        # output: (seq_len, batch_size, rnn_size)
        # reward: (batch_size,)
        batch_size = output.size(1)
        #aeq(batch_size, reward.size(0))
        scores = self.generator(self._bottle(output))
        gtruth = target.contiguous().view(-1)
        loss = self.criterion(scores, gtruth).view(-1, batch_size)  # (seq_len, batch_size)
        return loss, None
