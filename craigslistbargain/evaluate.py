import argparse
import torch.nn as nn
from torch import cuda
from onmt.Utils import use_gpu

from cocoa.io.utils import read_json, write_json, read_pickle, write_pickle, create_path
from cocoa.core.schema import Schema

from cocoa.neural.trainer import Trainer, Statistics
from cocoa.neural.loss import SimpleLossCompute
from cocoa.neural.beam import Scorer

from neural.utterance import UtteranceBuilder
from neural import get_data_generator, make_model_mappings
from neural import model_builder
from neural.evaluator import Evaluator
from neural.generator import get_generator
import options


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    options.add_data_generator_arguments(parser)
    options.add_generator_arguments(parser)
    args = parser.parse_args()

    # Know which arguments are for the models thus should not be
    # overwritten during test
    dummy_parser = argparse.ArgumentParser(description='duh')
    options.add_model_arguments(dummy_parser)
    options.add_data_generator_arguments(dummy_parser)
    dummy_args = dummy_parser.parse_known_args([])[0]

    if cuda.is_available() and not args.gpuid:
        print("WARNING: You have a CUDA device, should run with --gpuid 0")

    if args.gpuid:
        cuda.set_device(args.gpuid[0])

    # Load the model.
    mappings, model, model_args = \
        model_builder.load_test_model(args.checkpoint, args, dummy_args.__dict__)

    # Figure out src and tgt vocab
    make_model_mappings(model_args.model, mappings)

    schema = Schema(model_args.schema_path, None)
    data_generator = get_data_generator(args, model_args, schema, test=True)

    # Prefix: [GO, CATEGORY]
    # Just giving it GO seems okay as it can learn to copy the CATEGORY from the input
    scorer = Scorer(args.alpha)
    generator = get_generator(model, mappings['tgt_vocab'], scorer, args, model_args)
    builder = UtteranceBuilder(mappings['tgt_vocab'], args.n_best, has_tgt=True)
    evaluator = Evaluator(model, mappings, generator, builder, gt_prefix=1)
    evaluator.evaluate(args, model_args, data_generator)
