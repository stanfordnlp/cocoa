import time
import torch
import torch.nn as nn

from cocoa.neural.loss import SimpleLossCompute
from onmt.Loss import LossComputeBase
from onmt.Utils import aeq
import pdb
from symbols import markers
import random

class FBnegLossCompute(SimpleLossCompute):
    # Adds extra functionality to deal with the loss from selectors
    def __init__(self, generator, vocab):
        super(FBnegLossCompute, self).__init__(generator, vocab)
        self.selector = nn.LogSoftmax()
        self.select_criterion = nn.NLLLoss()
        self.vocab = vocab

    def compute_loss(self, targets, selections, output):
        # generator: Log softmax outputs to utterance vocab_size scores
        # decoder_outputs: (seq_len, batch_size, rnn_size)
        # output_decoder (13, 4, 256), bottled = (52, 256)
        scores = self.generator(self._bottle(output["decoder"]))
        # scores = (52 x 1246), (seq_len x vocab_size) since 1246 is vocab size
        gtruth = targets.contiguous().view(-1)
        # targets (13 x 4), so ground_truth = 52
        loss = self.criterion(scores, gtruth)
        loss_data = loss.data.clone()
        stats = self._stats(loss_data, scores.data, targets.view(-1).data)

        if random.random() < 0.003:
            seq_len, batch_size, hidden_dim = output["decoder"].shape
            print "loss:", loss.data.cpu().numpy()[0]
            # choose the 4th example after transposing 
            top = targets.transpose(0,1)[3].data.cpu().numpy()
            print " ".join([str(self.vocab.to_word(x)) for x in top if x != 1244])
            bottom = torch.max(scores.view(seq_len, batch_size, -1), dim=2)[1].data.cpu().numpy()
            # choose the 4th example
            print " ".join([str(self.vocab.to_word(x)) for x in bottom[:,3] if x != 1244])
            pdb.set_trace()

        '''
        # selector: GRU outputs to kb vocab_size scores/logprobs
        # output_selector (78, 4, 28), so bottled = (312, 28)
        select_scores = self.selector(self._bottle(output["selector"]))
        # since selector is just a softmax, shape stays the same at (312, 28)
        select_truth = selections.repeat(targets.shape[0], 1).contiguous().view(-1)
        # selections is a (batch_size=4, item_len=6) list, so repeat then flatten
        select_loss = self.select_criterion(select_scores, select_truth)
        select_data = select_loss.data.clone()
        select_stats = self._stats(select_data, select_scores.data,
                            select_truth.data)

        total_loss = loss + select_loss
        stats.update(select_stats)

        return total_loss, stats
        '''
        return loss, stats
