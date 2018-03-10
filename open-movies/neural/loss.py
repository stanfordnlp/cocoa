import torch
import pdb
import torch.nn as nn
from torch.autograd import Variable

class SimpleLossCompute(nn.Module):
    """
    Simpler Loss Computation class - does not perform Truncated BPTT,
        assumes we always normalize per sentence (rather than per word),
        removes label_smoothing, confidence-scores and sharding
    """
    def __init__(self, generator, vocab_size, padding_idx):
        super(SimpleLossCompute, self).__init__()
        self.generator = generator
        weight = torch.ones(vocab_size)
        weight[padding_idx] = 0
        self.criterion = nn.NLLLoss(weight)
        # self.criterion = nn.NLLLoss(weight, size_average=False)

    def simple_compute_loss(self, targets, outputs):
        loss = 0
        for idx, target in enumerate(targets):
            output = outputs[idx]
            loss += self.criterion(output, target)
        # loss.div(batch.size).backward()       # we don't need to divide by
        loss.backward()       # batch size since we set size_average to True

        return loss[0]

    def compute_accuracy(self, loss, targets, outputs, padding_idx):
        num_target_words, num_correct_words = 0,0
        for i, training_example in enumerate(targets):
            for j, correct_word in enumerate(training_example):
                pdb.set_trace()

                if correct_word != padding_idx:
                    num_target_words += 1
                    predicted_word = outputs[i][j]
                    if predicted_word == correct_word:
                        num_correct_words += 1

        return onmt.Statistics(loss, num_target_words, num_correct_words)
