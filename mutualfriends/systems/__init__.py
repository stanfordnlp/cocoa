from core.lexicon import Lexicon, add_lexicon_arguments
from core.inverse_lexicon import InverseLexicon, DefaultInverseLexicon
from rulebased_system import RulebasedSystem
from neural_system import NeuralSystem, add_neural_system_arguments
from cmd_system import CmdSystem

def add_system_arguments(parser):
    add_lexicon_arguments(parser)
    add_neural_system_arguments(parser)

def get_system(name, args, schema=None, timed=False):
    if name in ('rulebased', 'neural'):
        lexicon = Lexicon(schema, args.learned_lex, stop_words=args.stop_words, lexicon_path=args.lexicon)
        if args.inverse_lexicon:
            realizer = InverseLexicon.from_file(args.inverse_lexicon)
        else:
            realizer = DefaultInverseLexicon()

    if name == 'rulebased':
        return RulebasedSystem(lexicon, timed, realizer=realizer)
    elif name == 'neural':
        assert args.model_path
        return NeuralSystem(schema, lexicon, args.model_path, args.fact_check, args.decoding, realizer=realizer)
    elif name == 'cmd':
        return CmdSystem()
    else:
        raise ValueError('Unknown system %s' % name)

