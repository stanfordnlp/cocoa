import os
import argparse
from collections import namedtuple
from onmt.Utils import use_gpu

from cocoa.systems.system import System
from cocoa.sessions.timed_session import TimedSessionWrapper
from cocoa.core.util import read_pickle, read_json
from cocoa.neural.beam import Scorer

from neural.generator import get_generator
from sessions.neural_session import PytorchNeuralSession
from neural import model_builder, get_data_generator, make_model_mappings
from neural.preprocess import markers, TextIntMap, Preprocessor, Dialogue
from neural.batcher import DialogueBatcherFactory
from neural.utterance import UtteranceBuilder
import options


class PytorchNeuralSystem(System):
    """
    NeuralSystem loads a neural model from disk and provides a function instantiate a new dialogue agent (NeuralSession
    object) that makes use of this underlying model to send and receive messages in a dialogue.
    """
    def __init__(self, args, schema, price_tracker, model_path, timed):
        super(PytorchNeuralSystem, self).__init__()
        self.schema = schema
        self.price_tracker = price_tracker
        self.timed_session = timed

        # TODO: do we need the dummy parser?
        dummy_parser = argparse.ArgumentParser(description='duh')
        options.add_model_arguments(dummy_parser)
        options.add_data_generator_arguments(dummy_parser)
        dummy_args = dummy_parser.parse_known_args([])[0]

        # Load the model.
        mappings, model, model_args = model_builder.load_test_model(
                model_path, args, dummy_args.__dict__)
        self.model_name = model_args.model
        vocab = mappings['utterance_vocab']
        self.mappings = mappings

        generator = get_generator(model, vocab, Scorer(args.alpha), args, model_args)
        builder = UtteranceBuilder(vocab, args.n_best, has_tgt=True)

        preprocessor = Preprocessor(schema, price_tracker, model_args.entity_encoding_form,
                model_args.entity_decoding_form, model_args.entity_target_form)
        textint_map = TextIntMap(vocab, preprocessor)
        remove_symbols = map(vocab.to_ind, (markers.EOS, markers.PAD))
        use_cuda = use_gpu(args)

        kb_padding = mappings['kb_vocab'].to_ind(markers.PAD)
        dialogue_batcher = DialogueBatcherFactory.get_dialogue_batcher(model=self.model_name,
            kb_pad=kb_padding,
            mappings=mappings, num_context=model_args.num_context)

        # TODO: class variable is not a good way to do this
        Dialogue.preprocessor = preprocessor
        Dialogue.textint_map = textint_map
        Dialogue.mappings = mappings
        Dialogue.num_context = model_args.num_context

        Env = namedtuple('Env', ['model', 'vocab', 'preprocessor', 'textint_map',
            'stop_symbol', 'remove_symbols', 'gt_prefix',
            'max_len', 'dialogue_batcher', 'cuda',
            'dialogue_generator', 'utterance_builder', 'model_args'])
        self.env = Env(model, vocab, preprocessor, textint_map,
            stop_symbol=vocab.to_ind(markers.EOS), remove_symbols=remove_symbols,
            gt_prefix=1,
            max_len=20, dialogue_batcher=dialogue_batcher, cuda=use_cuda,
            dialogue_generator=generator, utterance_builder=builder, model_args=model_args)

    @classmethod
    def name(cls):
        return 'pt-neural'

    def new_session(self, agent, kb):
        if self.model_name in ('seq2seq', 'lf2lf'):
            session = PytorchNeuralSession(agent, kb, self.env)
        else:
            raise ValueError('Unknown model name {}'.format(self.model_name))
        if self.timed_session:
            session = TimedSessionWrapper(session)
        return session
