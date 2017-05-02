from src.basic.negotiation.price_tracker import PriceTracker
from rulebased_system import RulebasedSystem
from neural_system import NeuralSystem, add_neural_system_arguments
from cmd_system import CmdSystem

def add_system_arguments(parser):
    add_neural_system_arguments(parser)

def get_system(name, args, schema=None, timed=False):
    lexicon = PriceTracker()
    if name == 'rulebased':
        return RulebasedSystem(lexicon, timed)
    elif name == 'cmd':
        return CmdSystem()
    elif name == 'neural':
        assert args.model_path
        return NeuralSystem(schema, lexicon, args.model_path, args.decoding)
    else:
        raise ValueError('Unknown system %s' % name)
