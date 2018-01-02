from core.lexicon import Lexicon, add_lexicon_arguments
from model.generator import Templates, Generator
from model.manager import Manager
from rulebased_system import RulebasedSystem, add_rulebased_arguments
from cmd_system import CmdSystem

def add_system_arguments(parser):
    add_lexicon_arguments(parser)
    add_rulebased_arguments(parser)

def get_system(name, args, schema, model_path=None, timed=False):
    lexicon = Lexicon.from_pickle(args.lexicon)
    templates = Templates.from_pickle(args.templates)
    if name == 'rulebased':
        templates = Templates.from_pickle(args.templates)
        generator = Generator(templates)
        manager = Manager.from_pickle(args.policy)
        return RulebasedSystem(lexicon, generator, manager, timed)
    elif name == 'cmd':
        return CmdSystem()
    else:
        raise ValueError('Unknown system %s' % name)
