'''
Load data, train model and evaluate
'''

import argparse
import random
import os
import time as tm
from itertools import chain
import torch
import torch.nn as nn
from torch import cuda

from cocoa.io.utils import read_json, write_json, read_pickle, write_pickle, create_path
from cocoa.core.schema import Schema
from cocoa.neural.trainer import Statistics
from cocoa.neural.utterance import UtteranceBuilder
import cocoa.options

import onmt
from onmt.Utils import use_gpu

from neural.loss import SimpleLossCompute
from neural import get_data_generator, make_model_mappings
from neural import model_builder
from neural.trainer import Trainer
import options

def build_model(model_opt, opt, mappings, checkpoint):
    print 'Building model...'
    model = model_builder.make_base_model(model_opt, mappings,
                                    use_gpu(opt), checkpoint=checkpoint)

    if len(opt.gpuid) > 1:
        print('Multi gpu training: ', opt.gpuid)
        model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
    if opt.verbose:
        print(model)

    return model

def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)

def build_optim(opt, model, checkpoint):
    print('Making optimizer for training.')
    optim = onmt.Optim(
        opt.optim, opt.learning_rate, opt.max_grad_norm,
        model_size=opt.rnn_size)

    optim.set_parameters(model.parameters())

    return optim

def build_trainer(opt, model, mappings, optim):
    train_loss = make_loss(opt, model, mappings)
    valid_loss = make_loss(opt, model, mappings)
    trainer = Trainer(model, train_loss, valid_loss, optim)
    return trainer

def make_loss(opt, model, mappings):
    loss = SimpleLossCompute(model.generator, mappings['tgt_vocab'])
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    parser.add_argument('--stats-file', default='stats.json', help='Path to save json statistics (dataset, training etc.) file')
    parser.add_argument('--test', default=False, action='store_true', help='Test mode')
    parser.add_argument('--eval-output', default=None, help='JSON file to save evaluation results')
    parser.add_argument('--best', default=False, action='store_true', help='Test using the best model on dev set')
    parser.add_argument('--verbose', default=False, action='store_true', help='More prints')
    options.add_data_generator_arguments(parser)
    options.add_model_arguments(parser)
    cocoa.options.add_trainer_arguments(parser)
    args = parser.parse_args()

    random.seed(args.random_seed)
    model_args = args

    if torch.cuda.is_available() and not args.gpuid:
        print("WARNING: You have a CUDA device, should run with -gpuid 0")

    if args.gpuid:
        cuda.set_device(args.gpuid[0])
        if args.random_seed > 0:
            torch.cuda.manual_seed(args.random_seed)

    loading_timer = tm.time()

    schema = Schema(model_args.schema_path, None)
    data_generator = get_data_generator(args, model_args, schema)
    mappings = data_generator.mappings
    if args.verbose:
        print("Finished loading and pre-processing data, took {:.1f} seconds".format(tm.time() - loading_timer))

    # Preview a batch of data
    # train_data = data_generator.generator('train')
    # num_batches = train_data.next()
    # for i, batch_dialogue in enumerate(train_data):
    #    for batch in batch_dialogue['batch_seq']:
    #        data_generator.dialogue_batcher.print_batch(batch, i, data_generator.textint_map)
    #        import sys; sys.exit()

    # TODO: load from checkpoint
    ckpt = None

    # Build the model
    model = build_model(model_args, args, mappings, ckpt)
    tally_parameters(model)
    create_path(args.model_path)
    config_path = os.path.join(args.model_path, 'config.json')
    write_json(vars(args), config_path)

    builder = UtteranceBuilder(mappings['tgt_vocab'], 1, has_tgt=True)

    # Build optimizer and trainer
    optim = build_optim(args, model, ckpt)
    # vocab is used to make_loss, so use target vocab
    trainer = build_trainer(args, model, mappings, optim)
    trainer.builder = builder
    # Perform actual training
    trainer.learn(args, data_generator, report_func)
