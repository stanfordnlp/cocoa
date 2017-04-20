__author__ = 'anushabala'

import os
import argparse
import tensorflow as tf
from src.basic.systems.system import System
from src.basic.sessions.neural_session import RNNNeuralSession, GraphNeuralSession
from src.basic.sessions.timed_session import TimedSessionWrapper
from src.basic.util import read_pickle, read_json
from src.model.encdec import build_model
from src.model.preprocess import markers, TextIntMap, Preprocessor
from collections import namedtuple
from src.lib import logstats

def add_neural_system_arguments(parser):
    parser.add_argument('--decoding', nargs='+', default=['sample', 0], help='Decoding method')

class NeuralSystem(System):
    """
    NeuralSystem loads a neural model from disk and provides a function instantiate a new dialogue agent (NeuralSession
    object) that makes use of this underlying model to send and receive messages in a dialogue.
    """
    def __init__(self, schema, lexicon, model_path, decoding, timed_session=False, consecutive_entity=True, realizer=None):
        super(NeuralSystem, self).__init__()
        self.schema = schema
        self.lexicon = lexicon
        self.timed_session = timed_session
        # Whether to send two utterances containing entities consecutively
        # Don't do this because the bot can be too fast
        self.consecutive_entity = consecutive_entity

        # Load arguments
        args_path = os.path.join(model_path, 'config.json')
        config = read_json(args_path)
        config['batch_size'] = 1
        config['gpu'] = 0  # Don't need GPU for batch_size=1
        config['decoding'] = decoding
        args = argparse.Namespace(**config)

        mappings_path = os.path.join(model_path, 'vocab.pkl')
        mappings = read_pickle(mappings_path)
        vocab = mappings['vocab']

        args.dropout = 0
        logstats.add_args('model_args', args)
        model = build_model(schema, mappings, args)

        # Tensorflow config
        if args.gpu == 0:
            print 'GPU is disabled'
            config = tf.ConfigProto(device_count = {'GPU': 0})
        else:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5, allow_growth=True)
            config = tf.ConfigProto(device_count = {'GPU': 1}, gpu_options=gpu_options)

        # NOTE: need to close the session when done
        tf_session = tf.Session(config=config)
        tf.initialize_all_variables().run(session=tf_session)

        # Load TF model parameters
        ckpt = tf.train.get_checkpoint_state(model_path+'-best')
        assert ckpt, 'No checkpoint found'
        assert ckpt.model_checkpoint_path, 'No model path found in checkpoint'
        saver = tf.train.Saver()
        saver.restore(tf_session, ckpt.model_checkpoint_path)

        self.model_name = args.model
        if self.model_name == 'attn-copy-encdec':
            args.entity_target_form = 'graph'
            copy = True
        else:
            copy = False
        preprocessor = Preprocessor(schema, lexicon, args.entity_encoding_form, args.entity_decoding_form, args.entity_target_form)
        textint_map = TextIntMap(vocab, mappings['entity'], preprocessor)

        Env = namedtuple('Env', ['model', 'tf_session', 'preprocessor', 'vocab', 'copy', 'textint_map', 'stop_symbol', 'remove_symbols', 'max_len', 'consecutive_entity', 'realizer'])
        self.env = Env(model, tf_session, preprocessor, mappings['vocab'], copy, textint_map, stop_symbol=vocab.to_ind(markers.EOS), remove_symbols=map(vocab.to_ind, (markers.EOS, markers.PAD)), max_len=20, consecutive_entity=self.consecutive_entity, realizer=realizer)

    def __exit__(self, exc_type, exc_val, traceback):
        if self.tf_session:
            self.tf_session.close()

    @classmethod
    def name(cls):
        return 'neural'

    def new_session(self, agent, kb):
        if self.model_name == 'encdec':
            session = RNNNeuralSession(agent , kb, self.env)
        else:
            session = GraphNeuralSession(agent, kb, self.env)
        if self.timed_session:
            session = TimedSessionWrapper(agent, session)
	return session
