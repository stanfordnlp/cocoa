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

class FBnegLossCompute(SimpleLossCompute):
    # Adds extra functionality to deal with the loss from selectors
    def __init__(self, generator, tgt_vocab):
        super(FBnegLossCompute, self).__init__(generator, tgt_vocab)
        self.selector = nn.LogSoftmax()
        self.select_criterion = nn.NLLLoss()

    def compute_loss(self, targets, selections, output):
        # generator: Log softmax outputs to utterance vocab_size scores
        # scores output: (seq_len, batch_size, rnn_size)
        print("output decoder: {}".format(output["decoder"].shape))
        bd = self._bottle(output["decoder"]).shape
        print("bottled decoder: {}".format(bd))
        scores = self.generator(self._bottle(output["decoder"]))
        print("scores: {}".format(scores.shape))
        print("targets: {}".format(targets.shape))
        gtruth = targets.contiguous().view(-1)
        print("ground truth: {}".format(gtruth.shape))
        loss = self.criterion(scores, gtruth)
        loss_data = loss.data.clone()
        stats = self._stats(loss_data, scores.data, targets.view(-1).data)
        # selector: GRU outputs to kb vocab_size scores/logprobs
        # selections output: (seq_len=6x28, batch_size=16, rnn_size=64)
        print("output selector: {}".format(output["selector"].shape))
        bs = self._bottle(output["selector"]).shape
        print("bottled selector: {}".format(bs))
        select_scores = self.selector(self._bottle(output["selector"]))
        print("select scores: {}".format(select_scores.shape))
        print("selections: {} should match (6, 16, 28)".format(selections.shape))
        select_truth = selections.contiguous().view(-1)
        print("select truth: {}".format(select_truth.shape))
        select_loss = self.select_criterion(select_scores, select_truth)
        select_data = select_loss.data.clone()
        select_stats = self._stats(select_data, selections.data,
                            selections.view(-1).data)

        total_loss = loss + select_loss
        total_stats = stats.update(select_stats)

        return total_loss, total_stats

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
