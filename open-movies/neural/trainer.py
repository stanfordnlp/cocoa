from __future__ import division
from __future__ import print_function
"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
import time
import sys
import pdb # set_trace
import math
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.modules

from logging import Statistics, make_loss, report_func

def add_trainer_arguments(parser):
    group = parser.add_argument_group('Training')

    # Initialization
    group.add_argument('--pretrained-wordvec',
                       help="""If a valid path is specified, then this will load
                       pretrained word embeddings""")
    group.add_argument('--param-init', type=float, default=0.1,
                       help="""Parameters are initialized over uniform distribution
                       with support (-param_init, param_init).
                       Use 0 to not use initialization""")
    group.add_argument('--fix-pretrained-wordvec',
                       action='store_true',
                       help="Fix pretrained word embeddings.")
    group.add_argument('--train-from', default='', type=str,
                       help="""If training from a checkpoint then this is the
                       path to the pretrained model's state_dict.""")

    # Optimization
    group.add_argument('--batch-size', type=int, default=64,
                       help='Maximum batch size for training')
    group.add_argument('--epochs', type=int, default=14,
                       help='Number of training epochs')
    group.add_argument('--optim', default='sgd', help="""Optimization method.""",
                       choices=['sgd', 'adagrad', 'adadelta', 'adam'])
    group.add_argument('--max-grad-norm', type=float, default=5,
                       help="""If the norm of the gradient vector exceeds this,
                       renormalize it to have the norm equal to max_grad_norm""")
    group.add_argument('--dropout', type=float, default=0.3,
                       help="Dropout probability; applied in LSTM stacks.")
    group.add_argument('--learning-rate', type=float, default=1.0,
                       help="""Starting learning rate. Recommended settings:
                       sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")
    group.add_argument('-gpuid', default=[], nargs='+', type=int,
                       help="Use CUDA on the listed devices.")
    group.add_argument('-seed', type=int, default=-1,
                       help="""Random seed used for the experiments reproducibility.""")
    group.add_argument('--label-smoothing', type=float, default=0.0,
                       help="""Label smoothing value epsilon.
                       Probabilities of all non-true labels will be smoothed
                       by epsilon / (vocab_size - 1). Set to zero to turn off
                       label smoothing. For more detailed information, see:
                       https://arxiv.org/abs/1512.00567""")

    # Logging
    group.add_argument('--report-every', type=int, default=50,
                       help="Print stats at this interval.")
    group.add_argument('--model-filename', default='model',
                       help="""Model filename (the model will be saved as
                       <filename>_acc_ppl_e.pt where ACC is accuracy, PPL is
                       the perplexity and E is the epoch""")
    group.add_argument('--model-path', default='data/checkpoints',
                       help="""Which file the model checkpoints will be saved""")


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
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
    """

    def __init__(self, model, train_loss, valid_loss, optim, pad_id,
                 shard_size=32, data_type='text', norm_method="sents",
                 grad_accum_count=1):
        # Basic attributes.
        self.model = model
        self.pad_id = pad_id
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        print("shard size is 64: {}".format(shard_size) )
        self.shard_size = shard_size  # same as batch size, so no sharding occurs
        self.data_type = data_type
        self.norm_method = norm_method # by sentences vs. by tokens
        self.grad_accum_count = grad_accum_count

        assert(grad_accum_count > 0)

        # Set model in training mode.
        self.model.train()

    def learn(self, opt, model, data):
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
            train_iter = data.generator('train', opt.batch_size)
            train_stats = self.train_epoch(train_iter, epoch, report_func)
            print('Train perplexity: %g' % train_stats.ppl())

            # 2. Validate on the validation set.
            valid_iter = data.generator('dev', opt.batch_size)
            valid_stats = self.validate(valid_iter)
            print('Validation perplexity: %g' % valid_stats.ppl())

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


    def train_epoch(self, train_iter, epoch, report_func=None):
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
        batch_idx = 0
        true_batchs = []
        accum = 0
        normalization = 0
        # try:
        #     add_on = 0
        #     if len(train_iter) % self.grad_accum_count > 0:
        #         add_on += 1
        #     num_batches = len(train_iter) / self.grad_accum_count + add_on
        # except NotImplementedError:
            # Dynamic batching
            # num_batches = -1

        '''
        batch_dialogue is a dictionary with a key 'batch_seq'
        It's value is a list of dialogue-batches, in our case 5 batches
        Each batch of data has keys:
            'size': batch size, which is our case is 64
            'decoder_tokens': 64 lists of tokens
            'decoder_args': a dictionary with keys [inputs, targets, context]
                inputs and targets are len 64 arrays, that hold integer indexes
                of the words (as opposed to the text tokens)
            'agents': length 64 list referring to the speaker, either 0 or 1
            'uuids': length 64 list hold uuids for each example
            'encoder_tokens': same as decoder tokens, except for encoder
            'encoder_args': a dictionary with keys [inputs, context]
            'kbs': len 64 list of KB objects, the KB for open-movies simply holds
                a suggested topic and is non-critical, so it can be safely ignored
        '''
        for batch_dialogue in train_iter:
            for batch in batch_dialogue['batch_seq']:
                #cur_dataset = train_iter.get_cur_dataset()
                #self.train_loss.cur_dataset = cur_dataset
                # remove "make features"?
                # TODO: batch is a dict, needs to redo the interface with the model
                # they use a batch object and you call "batch.target", but ours is more raw
                # so we need to update the interface
                true_batchs.append(batch)
                accum += 1
                if self.norm_method == "tokens":
                    batch_indices = batch['decoder_args']['targets'].flatten()
                    non_PAD_tokens = sum([1 for bi in batch_indices if bi != self.pad_id])
                    normalization += non_PAD_tokens
                    self.non_pad_count = non_PAD_tokens
                else:
                    normalization += batch['size']

                pdb.set_trace()

                if accum == self.grad_accum_count:
                    self._gradient_accumulation(true_batchs, total_stats, report_stats)

                    if report_func is not None:
                        print("Using the reporting function.")
                        report_stats = report_func(epoch, batch_idx, num_batches,
                            total_stats.start_time, self.optim.lr, report_stats)

                    true_batchs = []
                    accum = 0
                    normalization = 0
                    batch_idx += 1

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

        for batch in valid_iter:
            encoder_inputs = batch['encoder_args']['inputs']
            decoder_inputs = batch['decoder_args']['inputs']
            decoder_targets = batch['decoder_args']['targets']

            src_lengths = [sum([1 for x in source if x != self.pad]) for source in encoder_inputs]
            outputs, attns, _ = self.model(encoder_inputs, decoder_inputs, src_lengths)
            val_loss = self.valid_loss.simple_compute_loss(targets, outputs)
            batch_stats = self.valid_loss.compute_stats(val_loss,
                                        targets, outputs, self.pad_id)
            stats.update(batch_stats)

        return stats

    def epoch_step(self, ppl, epoch):
        return self.optim.update_learning_rate(ppl, epoch)

    def drop_checkpoint(self, opt, epoch, valid_stats):
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
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.model_filename, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))

    def _gradient_accumulation(self, true_batchs, total_stats, report_stats):
        # if self.grad_accum_count > 1:
        #     self.model.zero_grad()
        for batch in true_batchs:
            dec_state = None
            encoder_inputs = batch['encoder_args']['inputs']
            decoder_inputs = batch['decoder_args']['inputs']
            decoder_targets = batch['decoder_args']['targets']
            # onmt.io.make_features(batch, 'tgt')
            # src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                src_lengths = [sum([1 for x in source if x != self.pad]) for source in encoder_inputs]
                report_stats.n_src_words += sum(src_lengths)
            else:
                src_lengths = None
            # target_size = batch['size']
            # for j in range(target_size):
            # 2. Forward-prop all but generator.
            # if self.grad_accum_count == 1:
            self.model.zero_grad()
            outputs, attns, dec_state = \
                self.model(encoder_inputs, decoder_inputs, src_lengths, dec_state)
            # 3. Compute loss
            loss_score = self.train_loss.simple_compute_loss(targets, outputs)
            batch_stats = self.train_loss.compute_accuracy(train_loss,
                                            targets, outputs, self.pad_id)
            # 4. Update the parameters and statistics.
            # if self.grad_accum_count == 1:
            self.optim.step()
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)
        # if self.grad_accum_count > 1:
        #     self.optim.step()