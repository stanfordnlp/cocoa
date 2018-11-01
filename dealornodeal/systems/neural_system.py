import os
import argparse
from collections import namedtuple
from onmt.Utils import use_gpu

from cocoa.systems.system import System
from cocoa.sessions.timed_session import TimedSessionWrapper
from cocoa.core.util import read_pickle, read_json
from cocoa.lib import logstats
from cocoa.neural.beam import Scorer

from fb_model import utils
from fb_model.agent import LstmRolloutAgent

from sessions.neural_session import FBNeuralSession, PytorchNeuralSession, PytorchLFNeuralSession
from neural import model_builder, get_data_generator, make_model_mappings
from neural.preprocess import markers, TextIntMap, Preprocessor, Dialogue
from neural.batcher import DialogueBatcherFactory
from neural.utterance import UtteranceBuilder
from neural.generator import get_generator
import options

# `args` for LstmRolloutAgent
Args = namedtuple('Args', ['temperature', 'domain'])

class FBNeuralSystem(System):
    def __init__(self, model_file, temperature, timed_session=False, gpu=False):
        super(FBNeuralSystem, self).__init__()
        self.timed_session = timed_session
        self.model = utils.load_model(model_file, gpu=gpu)
        self.args = Args(temperature=temperature, domain='object_division')

    @classmethod
    def name(cls):
        return 'neural'

    def new_session(self, agent, kb):
        session = FBNeuralSession(agent, kb, self.model, self.args)
        if self.timed_session:
            session = TimedSessionWrapper(session)
        return session

class PytorchNeuralSystem(System):
    def __init__(self, args, schema, lexicon, model_path, timed):
        super(PytorchNeuralSystem, self).__init__()
        self.schema = schema
        self.lexicon = lexicon
        self.timed_session = timed

        # TODO: do we need the dummy parser?
        dummy_parser = argparse.ArgumentParser(description='duh')
        options.add_model_arguments(dummy_parser)
        options.add_data_generator_arguments(dummy_parser)
        dummy_args = dummy_parser.parse_known_args([])[0]

        # Load the model.
        mappings, model, model_args = model_builder.load_test_model(
                model_path, args, dummy_args.__dict__)
        logstats.add_args('model_args', model_args)
        self.model_name = model_args.model
        utterance_vocab = mappings['utterance_vocab']
        kb_vocab = mappings['kb_vocab']
        self.mappings = mappings

        text_generator = get_generator(model, utterance_vocab, Scorer(args.alpha), args, model_args)
        builder = UtteranceBuilder(utterance_vocab, args.n_best, has_tgt=True)

        preprocessor = Preprocessor(schema, lexicon, model_args.entity_encoding_form,
                model_args.entity_decoding_form, model_args.entity_target_form)
        textint_map = TextIntMap(utterance_vocab, preprocessor)
        remove_symbols = map(utterance_vocab.to_ind, (markers.EOS, markers.PAD))
        use_cuda = use_gpu(args)

        dialogue_batcher = DialogueBatcherFactory.get_dialogue_batcher(model=self.model_name,
            kb_pad=None, mappings=mappings, num_context=model_args.num_context)

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
        known_models = ('seq2seq', 'lf2lf')
        if not self.model_name in known_models:
            raise ValueError('Unknown model name {}'.format(self.model_name))
        elif self.model_name == 'lf2lf':
            session = PytorchLFNeuralSession(agent, kb, self.env)
        else:
            session = PytorchNeuralSession(agent, kb, self.env)
        return session
