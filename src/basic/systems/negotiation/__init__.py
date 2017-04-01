from rulebased_system import RulebasedSystem
from src.basic.price_tracker import PriceTracker

def add_system_arguments(parser):
    return

def get_system(name, args, schema=None, timed=False):
    lexicon = PriceTracker()
    if name == 'rulebased':
        return RulebasedSystem(lexicon, timed)
    else:
        raise ValueError('Unknown system %s' % name)
