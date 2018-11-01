import time
import sys
import math
import numpy as np
import torch
import torch.nn as nn

from onmt.Utils import use_gpu

from cocoa.neural.trainer import Statistics
from cocoa.neural.trainer import Trainer as BaseTrainer
from cocoa.io.utils import create_path

class Trainer(BaseTrainer):
    ''' Class that controls the training process which inherits from Cocoa '''

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

    def _run_batch(self, batch, dec_state=None, enc_state=None):
        return self._run_seq2seq_batch(batch, dec_state, enc_state)

    def _run_seq2seq_batch(self, batch, dec_state=None, enc_state=None):
        encoder_inputs = batch.encoder_inputs
        decoder_inputs = batch.decoder_inputs
        targets = batch.targets
        lengths = batch.lengths
        #tgt_lengths = batch.tgt_lengths

        context_inputs = batch.context_inputs
        scene_inputs = batch.scene_inputs

        outputs, attns, dec_state = self.model(encoder_inputs,
                decoder_inputs, context_inputs, scene_inputs,
                lengths, dec_state, enc_state)

        return outputs, attns, dec_state

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

            outputs, attns, dec_state = self._run_batch(batch, None, enc_state)
            loss, batch_stats = self.train_loss.compute_loss(batch.targets, outputs)

            self.model.zero_grad()
            loss.backward()
            self.optim.step()

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # Don't backprop fully.
            if dec_state is not None:
                dec_state.detach()
