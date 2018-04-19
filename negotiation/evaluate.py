import argparse
import random
import os
import time
import pdb
from itertools import chain
import torch
import torch.nn as nn
from torch import cuda

from cocoa.io.utils import read_json, write_json, read_pickle, write_pickle, create_path
from cocoa.core.schema import Schema
from cocoa.lib import logstats

import onmt
from cocoa.pt_model.util import use_gpu

from neural.trainer import add_trainer_arguments, Trainer, Statistics
from neural.model_builder import add_model_arguments
from neural import add_data_generator_arguments, get_data_generator
from neural import model_builder
from neural.loss import SimpleLossCompute
from neural.evaluator import Evaluator, add_evaluator_arguments

#from model import add_data_generator_arguments, get_data_generator, add_model_arguments, build_model
#from model.learner import add_learner_arguments, get_learner
#from model.evaluate import get_evaluator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    parser.add_argument('--stats-file', help='Path to save json statistics (dataset, training etc.) file')
    add_data_generator_arguments(parser)
    add_evaluator_arguments(parser)
    args = parser.parse_args()

    # Know which arguments are for the models thus should not be
    # overwritten during test
    dummy_parser = argparse.ArgumentParser(description='duh')
    add_model_arguments(dummy_parser)
    add_data_generator_arguments(dummy_parser)
    dummy_args = dummy_parser.parse_known_args([])[0]

    if torch.cuda.is_available() and not args.gpuid:
        print("WARNING: You have a CUDA device, should run with --gpuid 0")

    if args.gpuid:
        cuda.set_device(args.gpuid[0])

    # Load the model.
    mappings, model, model_args = \
        model_builder.load_test_model(args, dummy_args.__dict__)

    # Figure out src and tgt vocab
    if model_args.model == 'seq2lf':
        mappings['src_vocab'] = mappings['vocab']
        mappings['tgt_vocab'] = mappings['lf_vocab']
    else:
        mappings['src_vocab'] = mappings['vocab']
        mappings['tgt_vocab'] = mappings['vocab']

    schema = Schema(model_args.schema_path, None)
    data_generator = get_data_generator(args, model_args, mappings, schema, test=True)

    # Prefix: [GO, CATEGORY]
    # Just giving it GO seems okay as it can learn to copy the CATEGORY from the input
    evaluator = Evaluator(model, mappings,  gt_prefix=1)
    evaluator.evaluate(args, model_args, data_generator)
