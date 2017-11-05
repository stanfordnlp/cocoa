from core.lexicon import Lexicon
from rulebased_system import RulebasedSystem
from cmd_system import CmdSystem
from neural_system import NeuralSystem, add_neural_system_arguments

def add_system_arguments(parser):
    parser.add_argument('--mappings', default='.', help='Directory to save mappings/vocab')
    add_neural_system_arguments(parser)

def get_system(name, args, schema=None, timed=False):
    lexicon = Lexicon(schema.values['item'])
    if name == 'rulebased':
        return RulebasedSystem(lexicon, timed)
    elif name == 'cmd':
        return CmdSystem()
    elif name == 'neural':
        return NeuralSystem(args.model_file, args.temperature, timed_session=timed, gpu=args.gpu)
    else:
        raise ValueError('Unknown system %s' % name)
