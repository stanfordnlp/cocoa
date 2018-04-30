import argparse
from collections import namedtuple

from system import System
from cocoa.core.util import read_pickle, read_json
from cocoa.pt_model.util import use_gpu
from cocoa.lib import logstats
from cocoa.sessions.timed_session import TimedSessionWrapper

from sessions.hybrid_session import HybridSession
from sessions.neural_session import PytorchNeuralSession

import torch
from neural import model_builder, get_data_generator, make_model_mappings
from neural.preprocess import markers, TextIntMap, Preprocessor, SpecialSymbols, Dialogue
from neural.batcher import DialogueBatcherFactory
from neural.evaluator import add_evaluator_arguments
from neural.beam import Scorer
#from neural.generator import Generator, Sampler
from neural.generator import get_generator
from neural.utterance import UtteranceBuilder


def add_hybrid_arguments(parser):
    parser.add_argument('--templates', help='Path to templates (.pkl)')
    parser.add_argument('--decoding', nargs='+', default=['sample', 0], help='Decoding method')
    parser.add_argument('--mappings', default='.', help='Directory to save mappings/vocab')
    parser.add_argument('--checkpoint', default='.', help='Directory to save learned models')
    add_evaluator_arguments(parser)

class HybridSystem(BaseRulebasedSystem):
    def __init__(self, lexicon, generator, timed_session):
        super(HybridSystem, self).__init__()
        self.timed_session = timed_session
        self.lexicon = lexicon
        self.generator = generator

    super(PytorchNeuralSystem, self).__init__()
        self.schema = schema
        self.price_tracker = price_tracker

        # Load config and set new args
        config_path = os.path.join(args.checkpoint, 'config.json')
        config = read_json(config_path)
        config['batch_size'] = 1
        config['gpu'] = 0  # Don't need GPU for batch_size=1
        config['dropout'] = 0  # Don't want dropout during test split
        config['decoding'] = args.decoding
        config['pretrained_wordvec'] = None
        # Merge config with existing options
        config_args = argparse.Namespace(**config)
        for arg in args.__dict__:
            if arg not in config_args:
                config_args.__dict__[arg] = args.__dict__[arg]
        # Assume that original arguments + config + model_builder options
        # includes all the args we need, so no need to create dummy_parser

        # Load the model.
        mappings, model, model_args = model_builder.load_test_model(
                model_path, args, config_args.__dict__)
        logstats.add_args('model_args', model_args)
        self.model_name = model_args.model
        vocab = mappings['utterance_vocab']
        self.mappings = mappings

        generator = get_generator(model, vocab, Scorer(args.alpha), args)
        builder = UtteranceBuilder(vocab, args.n_best, has_tgt=True)

        preprocessor = Preprocessor(schema, price_tracker, model_args.entity_encoding_form,
                model_args.entity_decoding_form, model_args.entity_target_form)
        textint_map = TextIntMap(vocab, preprocessor)
        remove_symbols = map(vocab.to_ind, (markers.EOS, markers.PAD))
        use_cuda = use_gpu(args)

        int_markers = SpecialSymbols(*[vocab.to_ind(m) for m in markers])
        kb_padding = mappings['kb_vocab'].to_ind(markers.PAD)
        dialogue_batcher = DialogueBatcherFactory.get_dialogue_batcher(self.model_name,
            int_markers=int_markers, slot_filling=False, kb_pad=kb_padding,
            mappings=mappings, num_context=model_args.num_context)

        #TODO: class variable is not a good way to do this
        Dialogue.preprocessor = preprocessor
        Dialogue.textint_map = textint_map
        Dialogue.mappings = mappings
        Dialogue.num_context = model_args.num_context

        Env = namedtuple('Env', ['model', 'vocab', 'preprocessor', 'textint_map',
            'stop_symbol', 'remove_symbols', 'gt_prefix',
            'max_len', 'dialogue_batcher', 'cuda',
            'dialogue_generator', 'utterance_builder'])
        self.env = Env(model, vocab, preprocessor, textint_map,
            stop_symbol=vocab.to_ind(markers.EOS), remove_symbols=remove_symbols,
            gt_prefix=1,
            max_len=20, dialogue_batcher=dialogue_batcher, cuda=use_cuda,
            dialogue_generator=generator, utterance_builder=builder)

    @classmethod
    def name(cls):
        return 'hybrid'

    def new_session(self, agent, kb, config=None):
        session = self._new_session(agent, kb, config)
        if self.timed_session:
            session = TimedSessionWrapper(session)
        return session

    def _new_session(self, agent, kb, config=None):
        session = PytorchNeuralSession(agent, kb, self.env, use_rl)
        return HybridSession.get_session(agent, kb, self.lexicon, config, self.generator)


