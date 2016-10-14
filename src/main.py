'''
Load data, learn model and evaluate
'''

import argparse
import random
import os
import time
import tensorflow as tf
from basic.util import read_json, write_json, read_pickle, write_pickle
from basic.dataset import add_dataset_arguments, read_dataset
from basic.schema import Schema
from basic.scenario_db import ScenarioDB, add_scenario_arguments
from basic.lexicon import Lexicon
from model.preprocess import DataGenerator
from model.encdec import add_model_arguments, EncoderDecoder, AttnEncoderDecoder, AttnCopyEncoderDecoder
from model.learner import add_learner_arguments, Learner
from model.graph import Graph, add_graph_arguments
from model.graph_embed import GraphEmbed, add_graph_embed_arguments
from model.kg_embed import add_kg_arguments, CBOWGraph
from lib import logstats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    parser.add_argument('--stats-file', help='Path to save json statistics (dataset, training etc.) file')
    parser.add_argument('--test', default=False, action='store_true', help='Test mode')
    parser.add_argument('--best', default=False, action='store_true', help='Test using the best model on dev set')
    parser.add_argument('--verbose', default=False, action='store_true', help='More prints')
    parser.add_argument('--learned-lex', default=False, action='store_true', help='if true have entity linking in lexicon use learned system')
    add_scenario_arguments(parser)
    add_dataset_arguments(parser)
    add_model_arguments(parser)
    add_graph_arguments(parser)
    add_graph_embed_arguments(parser)
    add_learner_arguments(parser)
    args = parser.parse_args()

    random.seed(args.random_seed)
    logstats.init(args.stats_file)
    logstats.add_args('config', args)

    schema = Schema(args.schema_path)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))
    lexicon = Lexicon(schema, args.learned_lex)

    dataset = read_dataset(scenario_db, args)

    # Save or load models
    if args.init_from:
        start = time.time()
        print 'Load model (config, vocab, checkpoint) from', args.init_from
        config_path = os.path.join(args.init_from, 'config.json')
        vocab_path = os.path.join(args.init_from, 'vocab.pkl')
        # Check config compatibility
        saved_config = read_json(config_path)
        curr_config = vars(args)
        assert_keys = ['model', 'rnn_size', 'rnn_type', 'num_layers']
        for k in assert_keys:
            assert saved_config[k] == curr_config[k], 'Command line arguments and saved arguments disagree on %s' % k

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
        mappings = None
        ckpt = None

    # Dataset
    use_kb = False if args.model == 'encdec' else True
    data_generator = DataGenerator(dataset.train_examples, dataset.test_examples, dataset.test_examples, schema, lexicon, mappings, use_kb)
    for d in data_generator.examples:
        logstats.add('data', d, 'num_examples', data_generator.num_examples[d])
    logstats.add('vocab_size', data_generator.vocab.size)

    # Save vocab
    if not mappings:
        mappings = {'vocab': data_generator.vocab,\
                    'entity': data_generator.entity_map,\
                    'relation': data_generator.relation_map}
        vocab_path = os.path.join(args.checkpoint, 'vocab.pkl')
        write_pickle(mappings, vocab_path)

    # Build the model
    vocab = mappings['vocab']
    tf.reset_default_graph()
    tf.set_random_seed(args.random_seed)
    kg = None
    if args.model == 'encdec':
        model = EncoderDecoder(vocab.size, args.rnn_size, args.rnn_type, args.num_layers)
    elif args.model.startswith('attn'):
        # Build graph embedding model
        max_degree = args.num_items + len(schema.attributes)
        Graph.static_init(schema, mappings['entity'], mappings['relation'], args.rnn_size, max_degree=max_degree, entity_hist_len=args.entity_hist_len)
        max_num_nodes = args.num_items * len(schema.attributes) * 2
        num_edge_labels = Graph.relation_map.size
        kg = GraphEmbed(max_num_nodes, num_edge_labels, args.node_embed_size, args.edge_embed_size, args.rnn_size, Graph.feat_size, entity_cache_size=args.entity_cache_size, mp_iter=args.mp_iter, message=args.combine_message, num_entities=Graph.entity_map.size, entity_embed_size=args.entity_embed_size, use_entity_embedding=args.use_entity_embedding)

        if args.model == 'attn-encdec':
            model = AttnEncoderDecoder(vocab, args.rnn_size, kg, args.rnn_type, args.num_layers, args.attn_scoring, args.attn_output)
        elif args.model == 'attn-copy-encdec':
            model = AttnCopyEncoderDecoder(vocab, args.rnn_size, kg, args.rnn_type, args.num_layers, args.attn_scoring, args.attn_output)
        else:
            raise ValueError('Unknown model')
    else:
        raise ValueError('Unknown model')

    # Tensorflow config
    if args.gpu == 0:
        print 'GPU is disabled'
        config = tf.ConfigProto(device_count = {'GPU': 0})
    else:
        config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.5

    learner = Learner(data_generator, model, args.verbose)
    if args.test:
        assert args.init_from and ckpt, 'No model to test'
        with tf.Session(config=config) as sess:
            tf.initialize_all_variables().run()
            print 'Load TF model'
            start = time.time()
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)
            print 'Done [%fs]' % (time.time() - start)
            bleu, ent_recall = learner.test_bleu(sess, 'test')
            print 'bleu=%.4f ent_recall=%.4f' % (bleu, ent_recall)
    else:
        learner.learn(args, config, ckpt)
