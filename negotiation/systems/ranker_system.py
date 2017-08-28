import os
import argparse
from collections import namedtuple
import tensorflow as tf

from cocoa.systems.system import System
from cocoa.sessions.timed_session import TimedSessionWrapper
from cocoa.core.util import read_pickle, read_json
from cocoa.lib import logstats

from sessions.ranker_session import IRRankerSession, NeuralRankerSession, StreamingDialogue
from model import build_model
from model.ranker import IRRanker, EncDecRanker
from model.preprocess import markers, TextIntMap, Preprocessor, SpecialSymbols
from model.batcher import DialogueBatcherFactory

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

class NeuralRankerSystem(System):
    def __init__(self, schema, price_tracker, retriever, model_path, mappings, timed_session=False):
        super(NeuralRankerSystem, self).__init__()
        self.schema = schema
        self.price_tracker = price_tracker
        self.timed_session = timed_session

        # Load arguments
        args_path = os.path.join(model_path, 'config.json')
        config = read_json(args_path)
        # TODO: handle this properly
        config['batch_size'] = 1
        config['pretrained_wordvec'] = None
        args = argparse.Namespace(**config)

        mappings_path = os.path.join(mappings, 'vocab.pkl')
        mappings = read_pickle(mappings_path)
        vocab = mappings['vocab']

        logstats.add_args('model_args', args)
        model = build_model(schema, mappings, None, args)

        # Tensorflow config
        if args.gpu == 0:
            print 'GPU is disabled'
            config = tf.ConfigProto(device_count = {'GPU': 0})
        else:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5, allow_growth=True)
            config = tf.ConfigProto(device_count = {'GPU': 1}, gpu_options=gpu_options)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

        int_markers = SpecialSymbols(*[mappings['vocab'].to_ind(m) for m in markers])
        model_config = {'retrieve': True}
        batcher = DialogueBatcherFactory.get_dialogue_batcher(model_config, int_markers=int_markers, slot_filling=False, kb_pad=mappings['kb_vocab'].to_ind(markers.PAD))

        StreamingDialogue.textint_map = textint_map
        StreamingDialogue.num_context = args.num_context
        StreamingDialogue.mappings = mappings

        Env = namedtuple('Env', ['ranker', 'retriever', 'tf_session', 'preprocessor', 'mappings', 'textint_map', 'batcher'])
        self.env = Env(model, retriever, tf_session, preprocessor, mappings, textint_map, batcher)

    @classmethod
    def name(cls):
        return 'ranker-neural'

    def new_session(self, agent, kb):
        session = NeuralRankerSession(agent, kb, self.env)
        if self.timed_session:
            session = TimedSessionWrapper(agent, session)
	return session

