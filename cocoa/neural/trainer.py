from __future__ import division

import time
import sys
import math
import numpy as np
import torch
import torch.nn as nn

from onmt.Trainer import Statistics as BaseStatistics
from onmt.Utils import use_gpu

from cocoa.io.utils import create_path


class Statistics(BaseStatistics):
    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; loss: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch, n_batches,
               self.mean_loss(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train

            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 data_type='text', norm_method="sents",
                 grad_accum_count=1, utterance_builder=None):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.data_type = data_type
        self.norm_method = norm_method # by sentences vs. by tokens
        self.grad_accum_count = grad_accum_count
        self.cuda = False
        self.best_valid_loss = None

        assert(grad_accum_count > 0)

        # Set model in training mode.
        self.model.train()

        # For debugging
        self.utterance_builder = utterance_builder

    def learn(self, opt, data, report_func):
        """Train model.
        Args:
            opt(namespace)
            model(Model)
            data(DataGenerator)
        """
        print('\nStart training...')
        print(' * number of epochs: %d' % opt.epochs)
        print(' * batch size: %d' % opt.batch_size)

        for epoch in range(opt.epochs):
            print('')

            # 1. Train for one epoch on the training set.
            train_iter = data.generator('train', cuda=use_gpu(opt))
            train_stats = self.train_epoch(train_iter, opt, epoch, report_func)
            print('Train loss: %g' % train_stats.mean_loss())

            # 2. Validate on the validation set.
            valid_iter = data.generator('dev', cuda=use_gpu(opt))
            valid_stats = self.validate(valid_iter)
            print('Validation loss: %g' % valid_stats.mean_loss())

            # 3. Log to remote server.
            #if opt.exp_host:
            #    train_stats.log("train", experiment, optim.lr)
            #    valid_stats.log("valid", experiment, optim.lr)
            #if opt.tensorboard:
            #    train_stats.log_tensorboard("train", writer, optim.lr, epoch)
            #    train_stats.log_tensorboard("valid", writer, optim.lr, epoch)

            # 4. Update the learning rate
            self.epoch_step(valid_stats.ppl(), epoch)

            # 5. Drop a checkpoint if needed.
            if epoch >= opt.start_checkpoint_at:
                self.drop_checkpoint(opt, epoch, valid_stats)


    def train_epoch(self, train_iter, opt, epoch, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        # Set model back to training mode.
        self.model.train()

        total_stats = Statistics()
        report_stats = Statistics()
        true_batchs = []
        accum = 0
        normalization = 0
        num_batches = train_iter.next()
        self.cuda = use_gpu(opt)

        for batch_idx, batch in enumerate(train_iter):
            true_batchs.append(batch)
            accum += 1

            if accum == self.grad_accum_count:
                self._gradient_accumulation(true_batchs, total_stats, report_stats)
                true_batchs = []
                accum = 0

            if report_func is not None:
                report_stats = report_func(opt, epoch, batch_idx, num_batches,
                    total_stats.start_time, report_stats)

        # Accumulate gradients one last time if there are any leftover batches
        # Should not run for us since we plan to accumulate gradients at every
        # batch, so true_batches should always equal candidate batches
        if len(true_batchs) > 0:
            self._gradient_accumulation(true_batchs, total_stats, report_stats)
            true_batchs = []

        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        num_val_batches = valid_iter.next()
        dec_state = None
        for batch in valid_iter:
            if batch is None:
                dec_state = None
                continue
            elif not self.model.stateful:
                dec_state = None
            enc_state = dec_state.hidden if dec_state is not None else None

            outputs, attns, dec_state = self._run_batch(batch, None, enc_state)
            _, batch_stats = self.valid_loss.compute_loss(batch.targets, outputs)
            stats.update(batch_stats)

        # Set model back to training mode
        self.model.train()

        return stats

    def epoch_step(self, ppl, epoch):
        return self.optim.update_learning_rate(ppl, epoch)

    def drop_checkpoint(self, opt, epoch, valid_stats, model_opt=None):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'opt': opt if not model_opt else model_opt,
            'epoch': epoch,
            'optim': self.optim,
        }
        path = self.checkpoint_path(epoch, opt, valid_stats)
        create_path(path)
        print 'Save checkpoint {path}'.format(path=path)
        torch.save(checkpoint, path)

        self.save_best_checkpoint(checkpoint, opt, valid_stats)

    def save_best_checkpoint(self, checkpoint, opt, valid_stats):
        if self.best_valid_loss is None or valid_stats.mean_loss() < self.best_valid_loss:
            self.best_valid_loss = valid_stats.mean_loss()
            path = '{root}/{model}_best.pt'.format(
                        root=opt.model_path,
                        model=opt.model_filename)

            print 'Save best checkpoint {path}'.format(path=path)
            torch.save(checkpoint, path)

    def checkpoint_path(self, epoch, opt, stats):
        path = '{root}/{model}_loss{loss:.2f}_e{epoch:d}.pt'.format(
                    root=opt.model_path,
                    model=opt.model_filename,
                    loss=stats.mean_loss(),
                    epoch=epoch)
        return path

    def _run_batch(self, batch, dec_state=None, enc_state=None):
        raise NotImplementedError

    def _gradient_accumulation(self, true_batchs, total_stats, report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        dec_state = None
        for batch in true_batchs:
            if batch is None:
                dec_state = None
                continue
            elif not self.model.stateful:
                dec_state = None
            enc_state = dec_state.hidden if dec_state is not None else None

            self.model.zero_grad()
            outputs, attns, dec_state = self._run_batch(batch, None, enc_state)

            loss, batch_stats = self.train_loss.compute_loss(batch.targets, outputs)
            loss.backward()
            self.optim.step()

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # Don't backprop fully.
            if dec_state is not None:
                dec_state.detach()
