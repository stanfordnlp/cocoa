__author__ = 'anushabala'

import os
import tensorflow as tf
from system import System
from src.basic.sessions.neural_session import NeuralSession
from src.basic.util import read_pickle, read_json
from model.encdec import build_model
from model.preprocess import EOT, EOS, TextIntMap
from collections import namedtuple

class NeuralSystem(System):
    """
    NeuralSystem loads a neural model from disk and provides a function instantiate a new dialogue agent (NeuralSession
    object) that makes use of this underlying model to send and receive messages in a dialogue.
    """
    def __init__(self, schema, lexicon, model_path):
        super(NeuralSystem, self).__init__()
        self.schema = schema
        self.lexicon = lexicon

        # Load arguments
        args_path = os.path.join(model_path, 'config.json')
        args = read_json(args_path)

        mappings_path = os.path.join(model_path, 'vocab.pkl')
        mappings = read_pickle(mappings_path)
        vocab = mappings['vocab']

        model = build_model(schema, mappings, args)

        # Load TF model parameters
        ckpt = tf.train.get_checkpoint_state(model_path)
        assert ckpt, 'No checkpoint found'
        assert ckpt.model_checkpoint_path, 'No model path found in checkpoint'
        # NOTE: need to close the session when done
        tf_session = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.tf_session, ckpt.model_checkpoint_path)

        copy = True if args.model == 'attn-copy-encdec' else False
        textint_map = TextIntMap(vocab, mappings['entity_map'], copy)

        preprocessor = Preprocessor(schema, lexicon)

        Env = namedtuple('Env', ['model', 'tf-session', 'preprocessor', 'vocab', 'copy', 'textint_map', 'stop_symbol', 'remove_symbols', 'max_len'])
        self.env = Env(model, tf_session, preprocessor, mappings['vocab'], copy, textint_map, stop_symbol=vocab.to_ind(EOT), remove_symbols=map(vocab.to_ind, (EOT, EOS)), max_len=20)

    def __exit__(self, exc_type, exc_val, traceback):
        if self.tf_session:
            self.tf_session.close()

    @classmethod
    def name(cls):
        return 'neural'

    def new_session(self, agent, kb):
        return NeuralSession(agent, kb, self.env)
