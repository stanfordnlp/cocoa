from core.split_tracker import SplitTracker
from rulebased_system import RulebasedSystem
from cmd_system import CmdSystem

def add_system_arguments(parser):
    # add_price_tracker_arguments(parser)
    parser.add_argument('--mappings', default='.', help='Directory to save mappings/vocab')

def get_system(name, args, schema=None, timed=False):
    tracker = SplitTracker()
    if name == 'rulebased':
        return RulebasedSystem(tracker, timed)
    elif name == 'cmd':
        return CmdSystem()
    else:
        raise ValueError('Unknown system %s' % name)
