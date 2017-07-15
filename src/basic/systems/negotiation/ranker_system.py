import os
import argparse
import tensorflow as tf
from src.basic.systems.system import System
from src.basic.sessions.negotiation.ranker_session import IRRankerSession, EncDecRankerSession
from src.basic.sessions.timed_session import TimedSessionWrapper
from src.basic.util import read_pickle, read_json
from src.model.negotiation import build_model
from src.model.negotiation.ranker import IRRanker, EncDecRanker
from src.model.negotiation.preprocess import markers, TextIntMap, Preprocessor
from collections import namedtuple
from src.lib import logstats

class IRRankerSystem(System):
    """
    RankerSystem loads a neural model from disk and provides a function instantiate a new dialogue agent (RankerSession
    object) that makes use of this underlying model to send and receive messages in a dialogue.
    """
    def __init__(self, schema, price_tracker, retriever, timed_session=False):
        super(IRRankerSystem, self).__init__()
        self.schema = schema
        self.price_tracker = price_tracker
        self.timed_session = timed_session

        Env = namedtuple('Env', ['ranker', 'retriever', 'preprocessor'])
        ranker = IRRanker()
        preprocessor = Preprocessor(schema, price_tracker, 'canonical', 'canonical', 'canonical')
        self.env = Env(ranker, retriever, preprocessor)

    @classmethod
    def name(cls):
        return 'ranker-ir'

    def new_session(self, agent, kb):
        session = IRRankerSession(agent, kb, self.env)
        if self.timed_session:
            session = TimedSessionWrapper(agent, session)
	return session

class EncDecRankerSystem(System):
    def __init__(self, schema, price_tracker, retriever, model_path, mappings, timed_session=False):
        super(EncDecRankerSystem, self).__init__()
        self.schema = schema
        self.price_tracker = price_tracker
        self.timed_session = timed_session

        # Load arguments
        args_path = os.path.join(model_path, 'config.json')
        config = read_json(args_path)
        # TODO: handle this properly
        config['batch_size'] = 1
        config['gpu'] = 0  # Don't need GPU for batch_size=1
        config['pretrained_wordvec'] = None
        args = argparse.Namespace(**config)

        mappings_path = os.path.join(mappings, 'vocab.pkl')
        mappings = read_pickle(mappings_path)
        vocab = mappings['vocab']

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
        tf_session.run(tf.global_variables_initializer())

        # Load TF model parameters
        ckpt = tf.train.get_checkpoint_state(model_path+'-best')
        assert ckpt, 'No checkpoint found'
        assert ckpt.model_checkpoint_path, 'No model path found in checkpoint'
        saver = tf.train.Saver()
        saver.restore(tf_session, ckpt.model_checkpoint_path)

        preprocessor = Preprocessor(schema, price_tracker, 'canonical', 'canonical', 'canonical')
        textint_map = TextIntMap(vocab, preprocessor)

        ranker = EncDecRanker(model)
        ranker.set_tf_session(tf_session)
        Env = namedtuple('Env', ['ranker', 'retriever', 'tf_session', 'preprocessor', 'mappings', 'textint_map'])
        self.env = Env(ranker, retriever, tf_session, preprocessor, mappings, textint_map)

    @classmethod
    def name(cls):
        return 'ranker-encdec'

    def new_session(self, agent, kb):
        session = EncDecRankerSession(agent, kb, self.env)
        if self.timed_session:
            session = TimedSessionWrapper(agent, session)
	return session

