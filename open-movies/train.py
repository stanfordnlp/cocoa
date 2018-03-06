'''
Load data, train model and evaluate
'''

import argparse
import random
import os
import time
from itertools import chain
import torch.nn as nn

from cocoa.io.utils import read_json, write_json, read_pickle, write_pickle, create_path
from cocoa.core.schema import Schema
from cocoa.lib import logstats

import onmt
from onmt.Utils import use_gpu

from neural.trainer import add_trainer_arguments, Trainer
from neural.model_builder import add_model_arguments
from neural import add_data_generator_arguments, get_data_generator
from neural import model_builder
from neural.loss import make_loss

#from model import add_data_generator_arguments, get_data_generator, add_model_arguments, build_model
#from model.learner import add_learner_arguments, get_learner
#from model.evaluate import get_evaluator

def build_model(model_opt, opt, checkpoint=None):
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
    if opt.train_from:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    else:
        print('Making optimizer for training.')
        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            model_size=opt.rnn_size)

    optim.set_parameters(model.parameters())

    return optim

def build_trainer(opt, model, mappings, optim):
    train_loss = make_loss(opt, mappings, model)
    valid_loss = make_loss(opt, mappings, model)
    trainer = Trainer(model, train_loss, valid_loss, optim)
    return trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    parser.add_argument('--stats-file', help='Path to save json statistics (dataset, training etc.) file')
    parser.add_argument('--test', default=False, action='store_true', help='Test mode')
    parser.add_argument('--eval', default=False, action='store_true', help='Eval mode')
    parser.add_argument('--eval-output', default=None, help='JSON file to save evaluation results')
    parser.add_argument('--best', default=False, action='store_true', help='Test using the best model on dev set')
    parser.add_argument('--verbose', default=False, action='store_true', help='More prints')
    add_data_generator_arguments(parser)
    add_model_arguments(parser)
    add_trainer_arguments(parser)
    args = parser.parse_args()

    random.seed(args.random_seed)
    if args.verbose:
        print("Stats file loaded from {}".format(args.stats_file))
    create_path(args.stats_file)
    logstats.init(args.stats_file)
    logstats.add_args('config', args)
    model_args = args
    ckpt = None

    # Load vocab
    # TODO: put this in DataGenerator
    vocab_path = os.path.join(model_args.mappings, 'vocab.pkl')
    if not os.path.exists(vocab_path):
        print 'Vocab not found at', vocab_path
        mappings = None
        args.ignore_cache = True
    else:
        if args.verbose:
            print 'Load vocab from', vocab_path
        mappings = read_pickle(vocab_path)
        for k, v in mappings.iteritems():
            print k, v.size

    schema = Schema(model_args.schema_path, None)

    data_generator = get_data_generator(args, model_args, mappings, schema)

    for d, n in data_generator.num_examples.iteritems():
        logstats.add('data', d, 'num_dialogues', n)

    # Save mappings
    if not mappings:
        mappings = data_generator.mappings
        vocab_path = os.path.join(args.mappings, 'vocab.pkl')
        write_pickle(mappings, vocab_path, ensure_path=True)
    for name, m in mappings.iteritems():
        logstats.add('mappings', name, 'size', m.size)

    # Preview a batch of data
    # train_data = data_generator.generator('train')
    # num_batches = train_data.next()
    # for i, batch_dialogue in enumerate(train_data):
    #    for batch in batch_dialogue['batch_seq']:
    #        data_generator.dialogue_batcher.print_batch(batch, i, data_generator.textint_map)
    #        import sys; sys.exit()

    logstats.add_args('model_args', model_args)

    # Build the model
    model = build_model(model_args, args, ckpt)
    tally_parameters(model)
    create_path(args.model_path)
    config_path = os.path.join(args.model_path, 'config.json')
    write_json(vars(args), config_path)

    # Build optimizer and trainer
    optim = build_optim(args, model, ckpt)
    trainer = build_trainer(args, model, mappings, optim)

    # Perform actual training
    trainer.train(args, model, data_generator)
