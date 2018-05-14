import time
import torch
import torch.nn as nn
import numpy as np

from cocoa.neural.loss import SimpleLossCompute
from onmt.Loss import LossComputeBase
from onmt.Utils import aeq
import pdb
from symbols import markers
import random

class FBnegLossCompute(SimpleLossCompute):
    # Adds extra functionality to deal with the loss from selectors
    def __init__(self, generator, vocab, model_type):
        tgt_vocab = vocab["utterance_vocab"]
        kb_vocab = vocab["kb_vocab"]
        super(FBnegLossCompute, self).__init__(generator, tgt_vocab)
        self.selector = nn.LogSoftmax()
        # self.utterance_criterion = nn.NLLLoss(weight*0.5, size_average=False)
        select_weight = torch.ones(kb_vocab.size) * 2
        select_weight[kb_vocab.to_ind(0)] = 0.5
        self.select_criterion = nn.NLLLoss(select_weight, size_average=False)
        self.vocab = vocab
        self.model_type = model_type

    def compute_loss(self, targets, selections, output):
        dec_out = output["decoder"] if self.model_type == 'seq_select' else output
        # generator: Log softmax outputs to utterance vocab_size scores
        # decoder_outputs: (seq_len, batch_size, rnn_size)
        # output_decoder (13, 4, 256), bottled = (52, 256)
        scores = self.generator(self._bottle(dec_out))
        # scores = (52 x 1246), (seq_len x vocab_size) since 1246 is vocab size
        utterance_truth = targets.contiguous().view(-1)
        # targets (13 x 4), so ground_truth = 52
        loss = self.criterion(scores, utterance_truth)
        loss_data = utterance_loss.data.clone()
        stats = self._stats(loss_data, scores.data, targets.view(-1).data)

        if self.model_type == 'seq_select':
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
            if random.random() < 0.003:
                print "utterance_loss:", utterance_loss.data.cpu().numpy()[0]
                print "select_loss:", select_loss.data.cpu().numpy()[0]
                print "select_truth:", select_truth.view(6,-1).transpose(0,1)
                print "select_scores:", torch.max(select_scores,1)[1].view(6,-1).transpose(0,1)

            '''
            For both losses, size_average is set to False, which means we sum
            the batch loss, rather than averaging it.  This is because the input
            of the loss function is resized from (seq_len, batch_size, vocab_size)
            into ((seq_len * batch_size), vocab_size), so averaging would divide
            by a factor of sequence length (which varies for each batch) and is
            thus uncertain.  Therefore, we later divide by batch size to normalize.'''
            loss += select_loss
            stats.update(select_stats)

        return loss, stats
