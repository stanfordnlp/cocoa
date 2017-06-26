'''
Load data, learn model and evaluate
'''

import argparse
import random
import os
import time
import tensorflow as tf
from itertools import chain
from src.basic.util import read_json, write_json, read_pickle, write_pickle
from src.basic.schema import Schema
from src.model.preprocess import add_data_generator_arguments, get_data_generator
from src.model.encdec import add_model_arguments, build_model
from src.model.learner import add_learner_arguments, Learner
from src.model.evaluate import Evaluator, LMEvaluator
# TODO
from src.model.negotiation.evaluate import CheatRetrievalEvaluator
from src.lib import logstats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    parser.add_argument('--stats-file', help='Path to save json statistics (dataset, training etc.) file')
    parser.add_argument('--test', default=False, action='store_true', help='Test mode')
    parser.add_argument('--cheat', default=False, action='store_true', help='Cheat mode')
    parser.add_argument('--best', default=False, action='store_true', help='Test using the best model on dev set')
    parser.add_argument('--verbose', default=False, action='store_true', help='More prints')
    parser.add_argument('--domain', type=str, choices=['MutualFriends', 'Matchmaking'])
    add_data_generator_arguments(parser)
    add_model_arguments(parser)
    add_learner_arguments(parser)
    args = parser.parse_args()

    random.seed(args.random_seed)
    logstats.init(args.stats_file)
    logstats.add_args('config', args)

    # Save or load models
    if args.init_from:
        start = time.time()
        print 'Load model (config, vocab, checkpoint) from', args.init_from
        config_path = os.path.join(args.init_from, 'config.json')
        vocab_path = os.path.join(args.init_from, 'vocab.pkl')
        saved_config = read_json(config_path)
        saved_config['decoding'] = args.decoding
        saved_config['batch_size'] = args.batch_size
        saved_config['pretrained_wordvec'] = None
        model_args = argparse.Namespace(**saved_config)

        # Checkpoint
        if args.test and args.best:
            ckpt = tf.train.get_checkpoint_state(args.init_from+'-best')
        else:
            ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, 'No checkpoint found'
        assert ckpt.model_checkpoint_path, 'No model path found in checkpoint'

        # Load vocab
        mappings = read_pickle(vocab_path)
        print 'Done [%fs]' % (time.time() - start)
    else:
        # TODO: factor. Process args
        if args.predict_price:
            # Output <price> and use predictor to fill in the number
            args.entity_decoding_form = 'type'
            args.entity_target_form = 'type'
        # Save config
        if not os.path.isdir(args.checkpoint):
            os.makedirs(args.checkpoint)
        config_path = os.path.join(args.checkpoint, 'config.json')
        write_json(vars(args), config_path)
        model_args = args
        mappings = None
        ckpt = None

    schema = Schema(model_args.schema_path, model_args.domain)

    data_generator = get_data_generator(args, model_args, mappings, schema)

    for d, n in data_generator.num_examples.iteritems():
        logstats.add('data', d, 'num_dialogues', n)

    # Save mappings
    if not mappings:
        mappings = data_generator.mappings
        vocab_path = os.path.join(args.checkpoint, 'vocab.pkl')
        write_pickle(mappings, vocab_path)
    for name, m in mappings.iteritems():
        logstats.add('mappings', name, 'size', m.size)

    # Build the model
    logstats.add_args('model_args', model_args)
    # TODO: return args as well; might be changed
    model = build_model(schema, mappings, model_args)

    # Tensorflow config
    if args.gpu == 0:
        print 'GPU is disabled'
        config = tf.ConfigProto(device_count = {'GPU': 0})
    else:
        print 'Using GPU'
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5, allow_growth=True)
        config = tf.ConfigProto(device_count = {'GPU': 1}, gpu_options=gpu_options)

    if args.cheat:
        responses = data_generator.get_all_responses('train')
        print 'All responses:', len(responses)
        evaluator = CheatRetrievalEvaluator(data_generator, None, responses, splits=('dev',), batch_size=args.batch_size, verbose=args.verbose)
        for split, test_data, num_batches in evaluator.dataset():
            print split, num_batches
            evaluator.test_response_generation(None, test_data, num_batches)
        import sys; sys.exit()

    # TODO:
    if args.model == 'lm':
        Evaluator = LMEvaluator

    if args.test:
        assert args.init_from and ckpt, 'No model to test'
        evaluator = Evaluator(data_generator, model, splits=('test',), batch_size=args.batch_size, verbose=args.verbose)
        learner = Learner(data_generator, model, evaluator, batch_size=args.batch_size, verbose=args.verbose, unconditional=args.unconditional)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            print 'Load TF model'
            start = time.time()
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)
            print 'Done [%fs]' % (time.time() - start)

            for split, test_data, num_batches in evaluator.dataset():
                results = learner.eval(sess, split, test_data, num_batches)
                learner.log_results(split, results)
    else:
        evaluator = Evaluator(data_generator, model, splits=('dev',), batch_size=args.batch_size, verbose=args.verbose)
        learner = Learner(data_generator, model, evaluator, batch_size=args.batch_size, verbose=args.verbose, unconditional=args.unconditional)
        learner.learn(args, config, args.stats_file, ckpt)
