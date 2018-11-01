from core.lexicon import Lexicon
from model.manager import Manager
from model.generator import Templates, Generator

from rulebased_system import RulebasedSystem
from cmd_system import CmdSystem
from neural_system import FBNeuralSystem, PytorchNeuralSystem
from hybrid_system import HybridSystem

def get_system(name, args, schema=None, timed=False, model_path=None):
    lexicon = Lexicon(schema.values['item'])
    if name == 'rulebased':
        templates = Templates.from_pickle(args.templates)
        generator = Generator(templates)
        manager = Manager.from_pickle(args.policy)
        return RulebasedSystem(lexicon, generator, manager, timed)
    elif name == 'hybrid':
        assert model_path
        templates = Templates.from_pickle(args.templates)
        manager = PytorchNeuralSystem(args, schema, lexicon, model_path, timed)
        generator = Generator(templates)
        return HybridSystem(lexicon, generator, manager, timed)
    elif name == 'cmd':
        return CmdSystem()
    elif name == 'fb-neural':
        assert model_path
        return FBNeuralSystem(model_path, args.temperature, timed_session=timed, gpu=False)
    elif name == 'pt-neural':
        assert model_path
        return PytorchNeuralSystem(args, schema, lexicon, model_path, timed)
    else:
        raise ValueError('Unknown system %s' % name)
