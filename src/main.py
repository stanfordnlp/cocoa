'''
Load data, learn model and evaluate
'''

import argparse
import random
import sys
from basic.util import read_json
from basic.dataset import add_dataset_arguments, read_dataset
from basic.schema import Schema
from basic.scenario_db import ScenarioDB, add_scenario_arguments
from basic.lexicon import Lexicon
from model.preprocess import DataGenerator
from model.encdec import add_model_arguments, EncoderDecoder
from model.learner import add_learner_arguments, Learner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    add_scenario_arguments(parser)
    add_dataset_arguments(parser)
    add_model_arguments(parser)
    add_learner_arguments(parser)
    args = parser.parse_args()

    random.seed(args.random_seed)

    schema = Schema(args.schema_path)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))
    lexicon = Lexicon(schema)

    dataset = read_dataset(scenario_db, args)
    data_generator = DataGenerator(dataset.train_examples, dataset.test_examples, None, lexicon)
    vocab = data_generator.vocab

    if args.model == 'encdec':
        model = EncoderDecoder(vocab.size, args.rnn_size, args.rnn_type, args.num_layers)
    else:
        raise ValueError('Unknown model')

    #data_generator.generator('train').next()
    #sys.exit(0)

    learner = Learner(data_generator, model)
    learner.learn(args)
