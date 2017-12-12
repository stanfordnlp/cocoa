import os
import argparse
from collections import namedtuple
import tensorflow as tf

from cocoa.systems.system import System
from cocoa.sessions.timed_session import TimedSessionWrapper
from cocoa.core.util import read_pickle, read_json
from cocoa.lib import logstats

from model import build_model
from model.retriever import Retriever, add_retriever_arguments
from model.preprocess import markers, TextIntMap, Preprocessor, SpecialSymbols, Dialogue
from model.batcher import DialogueBatcherFactory
from sessions.neural_session import GeneratorNeuralSession, SelectorNeuralSession

def add_neural_system_arguments(parser):
    parser.add_argument('--decoding', nargs='+', default=['sample', 0], help='Decoding method')
    parser.add_argument('--mappings', default='.', help='Directory to save mappings/vocab')
    parser.add_argument('--checkpoint', default='.', help='Directory to save learned models')
    add_retriever_arguments(parser)

class NeuralSystem(System):
    """
    NeuralSystem loads a neural model from disk and provides a function instantiate a new dialogue agent (NeuralSession
    object) that makes use of this underlying model to send and receive messages in a dialogue.
    """
    def __init__(self, schema, price_tracker, model_path, mappings_path, decoding, index=None, num_candidates=20, retriever_context_len=2, timed_session=False):
        super(NeuralSystem, self).__init__()
        self.schema = schema
        self.price_tracker = price_tracker
        self.timed_session = timed_session

        # Load arguments
        args_path = os.path.join(model_path, 'config.json')
        config = read_json(args_path)
        config['batch_size'] = 1
        config['gpu'] = 0  # Don't need GPU for batch_size=1
        config['decoding'] = decoding
        config['pretrained_wordvec'] = None
        args = argparse.Namespace(**config)

        vocab_path = os.path.join(mappings_path, 'vocab.pkl')
        mappings = read_pickle(vocab_path)
        vocab = mappings['vocab']

        # TODO: different models have the same key now
        args.dropout = 0
        logstats.add_args('model_args', args)
        model = build_model(schema, mappings, None, args)

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

        # Model config tells data generator which batcher to use
        model_config = {}
        if args.retrieve or args.model in ('ir', 'selector'):
            model_config['retrieve'] = True
        if args.predict_price:
            model_config['price'] = True

        self.model_name = args.model
        preprocessor = Preprocessor(schema, price_tracker, args.entity_encoding_form, args.entity_decoding_form, args.entity_target_form)
        textint_map = TextIntMap(vocab, preprocessor)
        int_markers = SpecialSymbols(*[mappings['vocab'].to_ind(m) for m in markers])
        dialogue_batcher = DialogueBatcherFactory.get_dialogue_batcher(model_config, int_markers=int_markers, slot_filling=False, kb_pad=mappings['kb_vocab'].to_ind(markers.PAD))

        # Retriever
        if args.model == 'selector':
            retriever = Retriever(index, context_size=retriever_context_len, num_candidates=num_candidates)
        else:
            retriever = None

        #TODO: class variable is not a good way to do this
        Dialogue.mappings = mappings
        Dialogue.textint_map = textint_map
        Dialogue.preprocessor = preprocessor
        Dialogue.num_context = args.num_context

        Env = namedtuple('Env', ['model', 'tf_session', 'preprocessor', 'vocab', 'textint_map', 'stop_symbol', 'remove_symbols', 'max_len', 'dialogue_batcher', 'retriever'])
        self.env = Env(model, tf_session, preprocessor, mappings['vocab'], textint_map, stop_symbol=vocab.to_ind(markers.EOS), remove_symbols=map(vocab.to_ind, (markers.EOS, markers.PAD)), max_len=20, dialogue_batcher=dialogue_batcher, retriever=retriever)

    def __exit__(self, exc_type, exc_val, traceback):
        if self.tf_session:
            self.tf_session.close()

    @classmethod
    def name(cls):
        return 'neural-{}'.format(self.model_name)

    def new_session(self, agent, kb):
        if self.model_name == 'encdec':
            session = GeneratorNeuralSession(agent , kb, self.env)
        elif self.model_name == 'selector':
            session = SelectorNeuralSession(agent , kb, self.env)
        else:
            raise ValueError('Unknown model name')
        if self.timed_session:
            session = TimedSessionWrapper(session)
	return session
