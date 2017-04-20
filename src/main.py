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
from src.basic.dataset import add_dataset_arguments, read_dataset
from src.basic.schema import Schema
from src.basic.scenario_db import ScenarioDB, add_scenario_arguments
from src.basic.lexicon import Lexicon, add_lexicon_arguments
from src.model.preprocess import DataGenerator, Preprocessor, add_preprocess_arguments
from src.model.encdec import add_model_arguments, build_model
from src.model.learner import add_learner_arguments, Learner
from src.model.evaluate import Evaluator
from src.model.graph import Graph, GraphMetadata, add_graph_arguments
from src.model.graph_embedder import add_graph_embed_arguments
from src.lib import logstats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    parser.add_argument('--stats-file', help='Path to save json statistics (dataset, training etc.) file')
    parser.add_argument('--test', default=False, action='store_true', help='Test mode')
    parser.add_argument('--best', default=False, action='store_true', help='Test using the best model on dev set')
    parser.add_argument('--verbose', default=False, action='store_true', help='More prints')
    parser.add_argument('--domain', type=str, choices=['MutualFriends', 'Matchmaking'])
    add_scenario_arguments(parser)
    add_lexicon_arguments(parser)
    add_dataset_arguments(parser)
    add_preprocess_arguments(parser)
    add_model_arguments(parser)
    add_graph_arguments(parser)
    add_graph_embed_arguments(parser)
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
        # Save config
        if not os.path.isdir(args.checkpoint):
            os.makedirs(args.checkpoint)
        config_path = os.path.join(args.checkpoint, 'config.json')
        write_json(vars(args), config_path)
        model_args = args
        mappings = None
        ckpt = None

    schema = Schema(model_args.schema_path, model_args.domain)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))
    dataset = read_dataset(scenario_db, args)
    print 'Building lexicon...'
    start = time.time()
    lexicon = Lexicon(schema, args.learned_lex, stop_words=args.stop_words)
    print '%.2f s'% (time.time() - start)

    # Dataset
    use_kb = False if model_args.model == 'encdec' else True
    copy = True if model_args.model == 'attn-copy-encdec' else False
    if model_args.model == 'attn-copy-encdec':
        model_args.entity_target_form = 'graph'
    preprocessor = Preprocessor(schema, lexicon, model_args.entity_encoding_form, model_args.entity_decoding_form, model_args.entity_target_form)
    if args.test:
        model_args.dropout = 0
        data_generator = DataGenerator(None, None, dataset.test_examples, preprocessor, schema, model_args.num_items, mappings, use_kb, copy)
    else:
        data_generator = DataGenerator(dataset.train_examples, dataset.test_examples, None, preprocessor, schema, model_args.num_items, mappings, use_kb, copy)
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
    model = build_model(schema, mappings, model_args)

    # Tensorflow config
    if args.gpu == 0:
        print 'GPU is disabled'
        config = tf.ConfigProto(device_count = {'GPU': 0})
    else:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5, allow_growth=True)
        config = tf.ConfigProto(device_count = {'GPU': 1}, gpu_options=gpu_options)

    if args.test:
        assert args.init_from and ckpt, 'No model to test'
        evaluator = Evaluator(data_generator, model, splits=('test',), batch_size=args.batch_size, verbose=args.verbose)
        learner = Learner(data_generator, model, evaluator, batch_size=args.batch_size, verbose=args.verbose)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            print 'Load TF model'
            start = time.time()
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)
            print 'Done [%fs]' % (time.time() - start)

            for split, test_data, num_batches in evaluator.dataset():
                print '================== Eval %s ==================' % split
                print '================== Sampling =================='
                start_time = time.time()
                bleu, (ent_prec, ent_recall, ent_f1) = evaluator.test_bleu(sess, test_data, num_batches)
                print 'bleu=%.4f/%.4f/%.4f entity_f1=%.4f/%.4f/%.4f time(s)=%.4f' % (bleu[0], bleu[1], bleu[2], ent_prec, ent_recall, ent_f1, time.time() - start_time)
                print '================== Perplexity =================='
                start_time = time.time()
                loss = learner.test_loss(sess, test_data, num_batches)
                print 'loss=%.4f time(s)=%.4f' % (loss, time.time() - start_time)
                logstats.add(split, {'bleu-4': bleu[0], 'bleu-3': bleu[1], 'bleu-2': bleu[2], 'entity_precision': ent_prec, 'entity_recall': ent_recall, 'entity_f1': ent_f1, 'loss': loss})
    else:
        evaluator = Evaluator(data_generator, model, splits=('dev',), batch_size=args.batch_size, verbose=args.verbose)
        learner = Learner(data_generator, model, evaluator, batch_size=args.batch_size, verbose=args.verbose)
        learner.learn(args, config, args.stats_file, ckpt)
