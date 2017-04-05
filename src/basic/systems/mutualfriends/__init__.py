from src.basic.lexicon import Lexicon, add_lexicon_arguments
from src.basic.inverse_lexicon import InverseLexicon
from heuristic_system import HeuristicSystem, add_heuristic_system_arguments
from rulebased_system import RulebasedSystem
from neural_system import NeuralSystem, add_neural_system_arguments
from cmd_system import CmdSystem

def add_system_arguments(parser):
    add_lexicon_arguments(parser)
    add_neural_system_arguments(parser)
    add_heuristic_system_arguments(parser)

def get_system(name, args, schema=None):
    lexicon = Lexicon(schema, args.learned_lex, stop_words=args.stop_words)
    if args.inverse_lexicon:
        realizer = InverseLexicon(schema, args.inverse_lexicon)
    else:
        realizer = None

    if name == 'rulebased':
        return RulebasedSystem(lexicon, realizer=realizer)
    elif name == 'heuristic':
        return HeuristicSystem(args.joint_facts, args.ask)
    elif name == 'neural':
        assert args.model_path
        return NeuralSystem(schema, lexicon, args.model_path, args.fact_check, args.decoding, realizer=realizer)
    elif name == 'cmd':
        return CmdSystem()
    else:
        raise ValueError('Unknown system %s' % name)

