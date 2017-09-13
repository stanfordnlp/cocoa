# from core.price_tracker import PriceTracker, add_price_tracker_arguments
from rulebased_system import RulebasedSystem
from cmd_system import CmdSystem

def add_system_arguments(parser):
    # add_price_tracker_arguments(parser)
    parser.add_argument('--mappings', default='.', help='Directory to save mappings/vocab')

def get_system(name, args, schema=None, timed=False):
    lexicon = None # PriceTracker(args.price_tracker_model)
    if name == 'rulebased':
        return RulebasedSystem(lexicon, timed)
    elif name == 'cmd':
        return CmdSystem()
    else:
        raise ValueError('Unknown system %s' % name)
