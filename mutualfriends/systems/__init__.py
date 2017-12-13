from core.lexicon import Lexicon, add_lexicon_arguments
from model.manager import Manager
from model.generator import Templates, Generator
from core.inverse_lexicon import InverseLexicon, DefaultInverseLexicon
from rulebased_system import RulebasedSystem, add_rulebased_arguments
from neural_system import NeuralSystem, add_neural_system_arguments
from cmd_system import CmdSystem

def add_system_arguments(parser):
    add_lexicon_arguments(parser)
    add_neural_system_arguments(parser)
    add_rulebased_arguments(parser)

def get_system(name, args, schema=None, timed=False, model_path=None):
    if name in ('rulebased', 'neural'):
        lexicon = Lexicon(schema, args.learned_lex, stop_words=args.stop_words, lexicon_path=args.lexicon)
        if args.inverse_lexicon:
            realizer = InverseLexicon.from_file(args.inverse_lexicon)
        else:
            realizer = DefaultInverseLexicon()
    if name == 'rulebased':
        templates = Templates.from_pickle(args.templates)
        generator = Generator(templates)
        manager = Manager.from_pickle(args.policy)
        return RulebasedSystem(lexicon, generator, manager, timed)
    elif name == 'neural':
        assert args.model_path
        return NeuralSystem(schema, lexicon, args.model_path, args.fact_check, args.decoding, realizer=realizer)
    elif name == 'cmd':
        return CmdSystem()
    else:
        raise ValueError('Unknown system %s' % name)

