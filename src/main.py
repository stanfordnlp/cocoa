'''
Load data, learn model and evaluate
'''

import argparse
import random
import os
import tensorflow as tf
from basic.util import read_json, write_json, read_pickle, write_pickle
from basic.dataset import add_dataset_arguments, read_dataset
from basic.schema import Schema
from basic.scenario_db import ScenarioDB, add_scenario_arguments
from basic.lexicon import Lexicon
from model.preprocess import DataGenerator
from model.encdec import add_model_arguments, EncoderDecoder, AttnEncoderDecoder
from model.learner import add_learner_arguments, Learner
from model.kg_embed import add_kg_arguments, CBOWGraph
from lib import logstats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    parser.add_argument('--stats-file', help='Path to save json statistics (dataset, training etc.) file')
    parser.add_argument('--test', default=False, action='store_true', help='Test mode')
    parser.add_argument('--verbose', default=False, action='store_true', help='More prints')
    add_scenario_arguments(parser)
    add_dataset_arguments(parser)
    add_model_arguments(parser)
    add_kg_arguments(parser)
    add_learner_arguments(parser)
    args = parser.parse_args()

    random.seed(args.random_seed)
    logstats.init(args.stats_file)
    logstats.add_args('config', args)

    schema = Schema(args.schema_path)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))
    lexicon = Lexicon(schema)

    dataset = read_dataset(scenario_db, args)

    # Save or load models
    if args.init_from:
        print 'Load model (config, vocab, checkpoing) from', args.init_from
        config_path = os.path.join(args.init_from, 'config.json')
        vocab_path = os.path.join(args.init_from, 'vocab.pkl')
        # Check config compatibility
        saved_config = read_json(config_path)
        curr_config = vars(args)
        assert_keys = ['model', 'rnn_size', 'rnn_type', 'num_layers']
        for k in assert_keys:
            assert saved_config[k] == curr_config[k], 'Command line arguments and saved arguments disagree on %s' % k

        # Checkpoint
        if args.test:
            ckpt = tf.train.get_checkpoint_state(args.init_from+'-best')
        else:
            ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, 'No checkpoint found'
        assert ckpt.model_checkpoint_path, 'No model path found in checkpoint'

        # Load vocab
        vocab = read_pickle(vocab_path)
    else:
        # Save config
        if not os.path.isdir(args.checkpoint):
            os.makedirs(args.checkpoint)
        config_path = os.path.join(args.checkpoint, 'config.json')
        write_json(vars(args), config_path)
        vocab = None
        ckpt = None

    data_generator = DataGenerator(dataset.train_examples, dataset.test_examples, dataset.test_examples, lexicon, vocab)
    # Dataset stats
    for d in data_generator.examples:
        logstats.add('data', d, 'num_examples', data_generator.num_examples
[d])
    logstats.add('vocab_size', data_generator.vocab.size)

    # Save vocab
    if not vocab:
        vocab = data_generator.vocab
        vocab_path = os.path.join(args.checkpoint, 'vocab.pkl')
        write_pickle(data_generator.vocab, vocab_path)

    # Build the graph
    if args.model == 'encdec':
        model = EncoderDecoder(vocab.size, args.rnn_size, args.rnn_type, args.num_layers)
    elif args.model == 'attn-encdec':
        if args.kg_model == 'cbow':
            kg = CBOWGraph(schema, args.kg_embed_size)
        else:
            raise ValueError('Unknown KG model')
        model = AttnEncoderDecoder(vocab.size, args.rnn_size, kg, args.rnn_type, args.num_layers)
    else:
        raise ValueError('Unknown model')

    # Tensorflow config
    if args.gpu == 0:
        print 'GPU is disabled'
        config = tf.ConfigProto(device_count = {'GPU': 0})
    else:
        config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    learner = Learner(data_generator, model, args.verbose)
    if args.test:
        assert args.init_from and ckpt, 'No model to test'
        with tf.Session(config=config) as sess:
            tf.initialize_all_variables().run()
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)
            bleu = learner.test_bleu(sess, 'test')
            print 'bleu=%.4f' % bleu
    else:
        learner.learn(args, config, ckpt)
