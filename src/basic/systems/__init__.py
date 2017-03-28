__author__ = 'anushabala'

import src.config as config
def add_base_system_arguments(parser):
    return

if config.task == config.MutualFriends:
    from src.basic.lexicon import Lexicon, add_lexicon_arguments
    from src.basic.inverse_lexicon import InverseLexicon
    from src.basic.systems.heuristic_system import HeuristicSystem, add_heuristic_system_arguments
    from src.basic.systems.rulebased_system import RulebasedSystem
    from src.basic.systems.neural_system import NeuralSystem, add_neural_system_arguments
    from src.basic.systems.cmd_system import CmdSystem

    def add_mutualfriends_system_arguments(parser):
        add_base_system_arguments(parser)
        add_lexicon_arguments(parser)
        add_neural_system_arguments(parser)
        add_heuristic_system_arguments(parser)
        parser.add_argument('--fact-check', default=False, action='store_true', help='Check if the utterance is true given the KB. Only work for simulated data.')

    def get_mutualfriends_system(args):
        lexicon = Lexicon(schema, args.learned_lex, stop_words=args.stop_words)
elif config.task == config.Negotiation:
    pass
else:
    raise ValueError('Unknown task: %s.' % config.task)



