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
        self.select_criterion = nn.NLLLoss(size_average=False)
        self.vocab = vocab

    def compute_loss(self, targets, selections, output):
        # generator: Log softmax outputs to utterance vocab_size scores
        # decoder_outputs: (seq_len, batch_size, rnn_size)
        # output_decoder (13, 4, 256), bottled = (52, 256)
        scores = self.generator(self._bottle(output["decoder"]))
        # scores = (52 x 1246), (seq_len x vocab_size) since 1246 is vocab size
        utterance_truth = targets.contiguous().view(-1)
        # targets (13 x 4), so ground_truth = 52
        utterance_loss = self.criterion(scores, utterance_truth)
        loss_data = utterance_loss.data.clone()
        stats = self._stats(loss_data, scores.data, targets.view(-1).data)
        
        # selector: GRU outputs to kb vocab_size scores/logprobs
        # output.selector (6, batch_size, kb_vocab_size) = (6, 16, 28)
        select_scores = self.selector(self._bottle(output["selector"]))
        # since selector is just a softmax, after bottling is ((6x16), 28) = (96,28)    
        select_truth = selections.contiguous().view(-1)
        # selections is now (item_len=6, batch_size=16) = 96 
        select_loss = self.select_criterion(select_scores, select_truth)
        # pdb.set_trace()
        select_data = select_loss.data.clone()
        select_stats = self._stats(select_data, select_scores.data,
                            select_truth.data)
        '''        
        if random.random() < 0.01:
            print "utterance_loss:", utterance_loss.data.cpu().numpy()[0]
            print "select_loss:", select_loss.data.cpu().numpy()[0]
            pdb.set_trace()

        For both losses, size_average is set to False, which means we sum
        the batch loss, rather than averaging it.  This is because the input
        of the loss function is resized from (seq_len, batch_size, vocab_size)
        into ((seq_len * batch_size), vocab_size), so averaging would divide
        by a factor of sequence length (which varies for each batch) and is
        thus uncertain.  Therefore, we later divide by batch size to normalize.'''
        loss = utterance_loss + select_loss
        stats.update(select_stats)

        return loss, stats
