import time
import torch
import pdb
import numpy as np
import torch.nn as nn
import math
import sys

from cocoa.pt_model.util import use_gpu
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
        numpy_targets = targets.data.numpy()
        numpy_outputs = outputs.data.numpy()

        for i, training_example in enumerate(numpy_targets):
            for j, correct_word in enumerate(training_example):
                if correct_word != padding_idx:
                    num_target_words += 1
                    predicted_word = np.argmax(numpy_outputs[i][j])
                    if predicted_word == correct_word:
                        num_correct_words += 1

        return Statistics(loss, num_target_words, num_correct_words)


def make_loss(opt, vocab_size, padding_idx, model):
    loss = SimpleLossCompute(model.generator, vocab_size, padding_idx)
    if use_gpu(opt):
        loss.cuda()
    return loss

def report_func(opt, epoch, batch, num_batches, start_time, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.
    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if (batch % opt.report_every) == (-1 % opt.report_every):
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        # reset the Statistics
        report_stats = Statistics()

    return report_stats

class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def ppl(self):
        if isinstance(self.loss, Variable):
            loss = self.loss.data.numpy()[0]
        entropy = min(loss / float(self.n_words), 100)

        perplexity = math.exp(entropy)
        return perplexity

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)

    def log_tensorboard(self, prefix, writer, lr, epoch):
        t = self.elapsed_time()
        values = {
            "ppl": self.ppl(),
            "accuracy": self.accuracy(),
            "tgtper": self.n_words / t,
            "lr": lr,
        }
        writer.add_scalars(prefix, values, epoch)

