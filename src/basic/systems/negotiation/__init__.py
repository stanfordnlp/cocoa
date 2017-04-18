from src.basic.price_tracker import PriceTracker
from rulebased_system import RulebasedSystem
from cmd_system import CmdSystem

def add_system_arguments(parser):
    return

def get_system(name, args, schema=None, timed=False):
    lexicon = PriceTracker()
    if name == 'rulebased':
        return RulebasedSystem(lexicon, timed)
    elif name == 'cmd':
        return CmdSystem()
    else:
        raise ValueError('Unknown system %s' % name)
