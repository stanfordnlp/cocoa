__author__ = 'anushabala'

import os
import tensorflow as tf
from system import System
from src.basic.sessions.neural_session import NeuralSession
from src.basic.util import read_pickle

class NeuralSystem(System):
    def __init__(self, schema, lexicon, model_path):
        super(NeuralSystem, self).__init__()

        # Load vocab
        vocab_path = os.path.join(model_path, 'vocab.pkl')
        vocab = read_pickle(vocab_path)

        # Load TF model (graph and parameters)
        ckpt = tf.train.get_checkpoint_state(model_path)
        assert ckpt, 'No checkpoint found'
        assert ckpt.model_checkpoint_path, 'No model path found in checkpoint'
        # NOTE: need to close the session when done
        self.tf_session = tf.Session()
        saver = tf.train.import_meta_graph('%s.meta' % ckpt.model_checkpoint_path)
        saver.restore(self.tf_session, ckpt.model_checkpoint_path)
        self.tf_graph = {}
        # TODO: add these nodes to collection during saving
        self.tf_graph['outputs'] = tf.get_collection('outputs')
        self.tf_graph['init_state'] = tf.get_collection('init_state')
        self.tf_graph['final_state'] = tf.get_collection('finalt_state')

        self.schema = schema
        self.lexicon = lexicon

    def __exit__(self, exc_type, exc_val, traceback):
        if self.tf_session:
            self.tf_session.close()

    def new_session(self, agent, kb):
        return NeuralSession(agent, kb, self.lexicon, self.vocab, self.tf_graph, self.tf_session)
