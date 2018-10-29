from cocoa.core.util import read_json, read_pickle

import options


def add_system_arguments(parser):
    # NOTE: hybrid system arguments are covered by neural system and rulebased system
    options.add_price_tracker_arguments(parser)
    options.add_neural_system_arguments(parser)
    options.add_rulebased_arguments(parser)

def get_system(name, args, schema=None, timed=False, model_path=None):
    from core.price_tracker import PriceTracker
    lexicon = PriceTracker(args.price_tracker_model)

    if name == 'rulebased':
        from rulebased_system import RulebasedSystem
        from model.generator import Templates, Generator
        from model.manager import Manager
        templates = Templates.from_pickle(args.templates)
        generator = Generator(templates)
        manager = Manager.from_pickle(args.policy)
        return RulebasedSystem(lexicon, generator, manager, timed)
    elif name == 'hybrid':
        from hybrid_system import HybridSystem
        templates = Templates.from_pickle(args.templates)
        manager = PytorchNeuralSystem(args, schema, lexicon, model_path, timed)
        generator = Generator(templates)
        return HybridSystem(lexicon, generator, manager, timed)
    elif name == 'cmd':
        from cmd_system import CmdSystem
        return CmdSystem()
    elif name == 'pt-neural':
        from neural_system import PytorchNeuralSystem
        assert model_path
        return PytorchNeuralSystem(args, schema, lexicon, model_path, timed)
    else:
        raise ValueError('Unknown system %s' % name)
