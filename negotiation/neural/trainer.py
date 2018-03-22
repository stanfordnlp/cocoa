from __future__ import division
#from __future__ import print_function
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
#import onmt.modules
from onmt.Trainer import Statistics as BaseStatistics

from cocoa.pt_model.util import smart_variable, basic_variable, use_gpu


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
    # group.add_argument('--batches_per_epoch', type=int, default=10,
    #                    help='Data comes from a generator, which is unlimited, so we need to set some artificial limit.')
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
    group.add_argument('--gpuid', default=[], nargs='+', type=int,
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
    group.add_argument('--report-every', type=int, default=5, choices=[1,10],
                       help="Print stats at this many batch intervals")
    group.add_argument('--model-filename', default='model',
                       help="""Model filename (the model will be saved as
                       <filename>_acc_ppl_e.pt where ACC is accuracy, PPL is
                       the perplexity and E is the epoch""")
    group.add_argument('--model-path', default='data/checkpoints',
                       help="""Which file the model checkpoints will be saved""")


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
              (epoch, batch,  n_batches,
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
        self.shard_size = shard_size  # same as batch size, so no sharding occurs
        self.data_type = data_type
        self.norm_method = norm_method # by sentences vs. by tokens
        self.grad_accum_count = grad_accum_count

        assert(grad_accum_count > 0)

        # Set model in training mode.
        self.model.train()

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
            train_iter = data.generator('train')
            length_train_batches = train_iter.next()  # not sure why this is needed, see 'generator' in preprocess.py
            train_stats = self.train_epoch(train_iter, opt, epoch, report_func)
            print('Train perplexity: %g' % train_stats.ppl())

            # 2. Validate on the validation set.
            valid_iter = data.generator('dev', opt.batch_size)
            length_val_batches = valid_iter.next()
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
        batch_idx = 0
        normalization = 0
        num_batches = -1
        cuda = use_gpu(opt)

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
        ## for batch_idx in range(opt.batches_per_epoch):
        #epoch_data = train_iter.next()
        #num_batches = len(epoch_data['batch_seq'])   # number of batches per epoch, given from the dataset
        idx = 0
        for dialogue_batch in train_iter:
            for batch in dialogue_batch['batch_seq']:
                true_batchs.append(batch)
                accum += 1
                if self.norm_method == "tokens":
                    batch_indices = batch['decoder_args']['targets'].flatten()
                    non_PAD_tokens = sum([1 for bi in batch_indices if bi != self.pad_id])
                    normalization += non_PAD_tokens
                    self.non_pad_count = non_PAD_tokens
                else:
                    normalization += batch['size']

                if accum == self.grad_accum_count:
                    self._gradient_accumulation(true_batchs, total_stats, report_stats, cuda)

                    if report_func is not None:
                        report_stats = report_func(opt, epoch, idx, num_batches,
                            total_stats.start_time, report_stats)

                    true_batchs = []
                    accum = 0
                    normalization = 0
                    idx += 1

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

            src_lengths = [sum([1 for x in source if x != self.pad_id]) for source in encoder_inputs]
            outputs, attns, _ = self.model(encoder_inputs, decoder_inputs, src_lengths)
            val_loss, batch_stats = self.valid_loss.compute_loss(targets, outputs)
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

    def _gradient_accumulation(self, true_batchs, total_stats, report_stats, cuda):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            dec_state = None
            encoder_inputs = batch['encoder_args']['inputs']
            # src = onmt.io.make_features(batch, 'src', self.data_type)
            # onmt.io.make_features(batch, 'tgt')
            #if self.data_type == 'text':
            #    src_lengths = [sum([1 for x in source if x != self.pad_id]) for source in encoder_inputs]
            #    report_stats.n_src_words += sum(src_lengths)
            #else:
            #    src_lengths = None

            encoder_inputs = self.prepare_data(encoder_inputs, cuda)
            decoder_inputs = self.prepare_data(batch['decoder_args']['inputs'], cuda)
            targets = self.prepare_data(batch['decoder_args']['targets'], cuda)
            lengths = batch['encoder_args']['lengths']

            # 2. Forward-prop all but generator.
            #if self.grad_accum_count == 1:
            self.model.zero_grad()

            outputs, attns, dec_state = \
                self.model(encoder_inputs, decoder_inputs, lengths, dec_state)

            # 3. Compute loss
            loss, batch_stats = self.train_loss.compute_loss(targets, outputs)
            loss.backward()
            self.optim.step()

            # 4. Update the parameters and statistics.
            #if self.grad_accum_count == 1:
            #batch_stats = self.train_loss.compute_accuracy(loss_score,
            #                                targets, outputs, self.pad_id)
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # If truncated, don't backprop fully.
            if dec_state is not None:
                dec_state.detach()

    def prepare_data(self, data, cuda):
        #print data.shape
        #print data
        result = smart_variable(data.tolist(), "list", cuda)
        #result = torch.from_numpy(data)
        result = result.transpose(0,1)  # change into (seq_len, batch_size)
        return result
