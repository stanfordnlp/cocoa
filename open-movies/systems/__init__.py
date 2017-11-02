from core.lexicon import Lexicon, add_lexicon_arguments
from model.templates import Templates
from rulebased_system import RulebasedSystem
from cmd_system import CmdSystem

def add_system_arguments(parser):
    parser.add_argument('--templates', help='Path to pickled templates')
    parser.add_argument('--lexicon', help='Path to pickled lexicon')
    add_lexicon_arguments(parser)

def get_system(name, args, schema=None, timed=False):
  lexicon = Lexicon.from_pickle(args.lexicon)
  templates = Templates.from_pickle(args.templates)
  if name == 'rulebased':
      return RulebasedSystem(lexicon, templates, timed)
  elif name == 'cmd':
      return CmdSystem()
  else:
      raise ValueError('Unknown system %s' % name)
