from core.lexicon import Lexicon
from model.manager import Manager
from model.generator import Templates, Generator
from rulebased_system import RulebasedSystem, add_rulebased_arguments
from cmd_system import CmdSystem
from neural_system import NeuralSystem, add_neural_system_arguments

def add_system_arguments(parser):
    parser.add_argument('--mappings', default='.', help='Directory to save mappings/vocab')
    add_neural_system_arguments(parser)
    add_rulebased_arguments(parser)

def get_system(name, args, schema=None, timed=False, model_path=None):
    lexicon = Lexicon(schema.values['item'])
    if name == 'rulebased':
        templates = Templates.from_pickle(args.templates)
        generator = Generator(templates)
        manager = Manager.from_pickle(args.policy)
        return RulebasedSystem(lexicon, generator, manager, timed)
    elif name == 'cmd':
        return CmdSystem()
    elif name == 'neural':
        return NeuralSystem(model_path, args.temperature, timed_session=timed, gpu=args.gpu)
    else:
        raise ValueError('Unknown system %s' % name)
