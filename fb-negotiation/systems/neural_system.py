import os
import argparse
from collections import namedtuple

from cocoa.systems.system import System
from cocoa.sessions.timed_session import TimedSessionWrapper
from cocoa.core.util import read_pickle, read_json
from cocoa.pt_model.util import use_gpu
from cocoa.lib import logstats
from cocoa.neural.beam import Scorer
from neural.generator import FBnegSampler

from fb_model import utils
from fb_model.agent import LstmRolloutAgent

from sessions.neural_session import PytorchNeuralSession
from neural import model_builder, get_data_generator, make_model_mappings
from neural.preprocess import markers, TextIntMap, Preprocessor, Dialogue
from neural.batcher import DialogueBatcherFactory
from neural.evaluator import add_evaluator_arguments
from neural.utterance import UtteranceBuilder

def add_neural_system_arguments(parser):
    parser.add_argument('--decoding', nargs='+', default=['sample', 0],
        help='Decoding method describing whether or not to sample')
    parser.add_argument('--temperature', type=float, default=1.0,
        help='a float from 0.0 to 1.0 for how to vary sampling')
    parser.add_argument('--checkpoint', type=str, help='model file folder')
    add_evaluator_arguments(parser)
    #parser.add_argument('--num_types', type=int, default=3,
    #    help='number of object types')
    #parser.add_argument('--num_objects', type=int, default=6,
    #    help='total number of objects')
    #parser.add_argument('--max_score', type=int, default=10,
    #    help='max score per object')
    #parser.add_argument('--score_threshold', type=int, default=6,
    #    help='successful dialog should have more than score_threshold in score')

# `args` for LstmRolloutAgent
Args = namedtuple('Args', ['temperature', 'domain'])

class NeuralSystem(System):
    def __init__(self, schema, lexicon, model_path, mappings_path, decoding,
            index=None, num_candidates=20, context_len=2, timed_session=False):
        super(NeuralSystem, self).__init__()
        self.schema
        self.lexicon = lexicon
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
        vocab = mappings['utterance_vocab']

        # TODO: different models have the same key now
        args.dropout = 0
        logstats.add_args('model_args', args)
        self.model = utils.load_model(model_path, gpu=gpu)
        self.args = Args(temperature=temperature, domain='object_division')

        preprocessor = Preprocessor(schema, lexicon, args.entity_encoding_form, args.entity_decoding_form, args.entity_target_form)
        textint_map = TextIntMap(vocab, preprocessor)
        dialogue_batcher = DialogueBatcherFactory.get_dialogue_batcher(model_config, slot_filling=False, kb_pad=mappings['kb_vocab'].to_ind(markers.PAD), mappings=mappings)
        #TODO: class variable is not a good way to do this
        Dialogue.mappings = mappings
        Dialogue.textint_map = textint_map
        Dialogue.preprocessor = preprocessor
        Dialogue.num_context = args.num_context

        Env = namedtuple('Env', ['model', 'tf_session', 'preprocessor', 'vocab', 'textint_map', 'stop_symbol', 'remove_symbols', 'max_len', 'dialogue_batcher', 'retriever'])
        self.env = Env(model, tf_session, preprocessor, mappings['utterance_vocab'], textint_map, stop_symbol=vocab.to_ind(markers.EOS), remove_symbols=map(vocab.to_ind, (markers.EOS, markers.PAD)), max_len=20, dialogue_batcher=dialogue_batcher, retriever=retriever)


    @classmethod
    def name(cls):
        return 'neural'

    def new_session(self, agent, kb):
        session = NeuralSession(agent, kb, self.model, self.args)
        if self.timed_session:
            session = TimedSessionWrapper(session)
        return session


class PytorchNeuralSystem(System):
    def __init__(self, args, schema, lexicon, model_path, timed):
        super(PytorchNeuralSystem, self).__init__()
        self.schema = schema
        self.lexicon = lexicon
        self.timed_session = timed

        # Load config and set new args
        config_path = os.path.join(args.checkpoint, 'config.json')
        config = read_json(config_path)
        config['batch_size'] = 1
        config['gpu'] = 0  # Don't need GPU for batch_size=1
        config['dropout'] = 0  # Don't want dropout during test split
        config['decoding'] = args.decoding
        # Merge config with existing options
        config_args = argparse.Namespace(**config)
        for arg in args.__dict__:
            if arg not in config_args:
                config_args.__dict__[arg] = args.__dict__[arg]

        # Load the model.
        mappings, model, model_args = model_builder.load_test_model(
                model_path, args, config_args.__dict__)
        logstats.add_args('model_args', model_args)
        self.model_name = model_args.model
        utterance_vocab = mappings['utterance_vocab']
        kb_vocab = mappings['kb_vocab']
        self.mappings = mappings

        text_generator = FBnegSampler(model, utterance_vocab, config_args.temperature, 
                args.max_length, use_gpu(args))
        builder = UtteranceBuilder(utterance_vocab, args.n_best, has_tgt=True)

        preprocessor = Preprocessor(schema, lexicon, model_args.entity_encoding_form,
                model_args.entity_decoding_form, model_args.entity_target_form)
        textint_map = TextIntMap(utterance_vocab, preprocessor)
        remove_symbols = map(utterance_vocab.to_ind, (markers.EOS, markers.PAD))
        use_cuda = use_gpu(args)

        kb_padding = mappings['kb_vocab'].to_ind(markers.PAD)
        dialogue_batcher = DialogueBatcherFactory.get_dialogue_batcher(model=self.model_name,
            kb_pad=kb_padding, mappings=mappings, num_context=model_args.num_context)

        #TODO: class variable is not a good way to do this
        Dialogue.preprocessor = preprocessor
        Dialogue.textint_map = textint_map
        Dialogue.mappings = mappings
        Dialogue.num_context = model_args.num_context

        Env = namedtuple('Env', ['model', 'utterance_vocab', 'kb_vocab', 
            'preprocessor', 'textint_map', 'stop_symbol', 
            'remove_symbols', 'gt_prefix',
            'max_len', 'dialogue_batcher', 'cuda',
            'dialogue_generator', 'utterance_builder', 'model_args'])
        self.env = Env(model, utterance_vocab, kb_vocab, 
            preprocessor, textint_map, stop_symbol=utterance_vocab.to_ind(markers.EOS), 
            remove_symbols=remove_symbols, gt_prefix=1,
            max_len=20, dialogue_batcher=dialogue_batcher, cuda=use_cuda,
            dialogue_generator=text_generator, utterance_builder=builder, model_args=model_args)

    @classmethod
    def name(cls):
        return 'pt-neural'

    def new_session(self, agent, kb):
        if self.model_name in ('seq2seq', 'seq2lf', 'sum2sum', 'sum2seq', 'lf2lf', 'lflm'):
            session = PytorchNeuralSession(agent, kb, self.env)
        else:
            raise ValueError('Unknown model name {}'.format(self.model_name))
        return session
