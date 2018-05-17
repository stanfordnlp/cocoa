from cocoa.core.util import read_json, read_pickle

from core.price_tracker import PriceTracker, add_price_tracker_arguments
from rulebased_system import RulebasedSystem, add_rulebased_arguments
from configurable_rulebased_system import ConfigurableRulebasedSystem, add_configurable_rulebased_arguments
from neural_system import NeuralSystem, add_neural_system_arguments, PytorchNeuralSystem
from cmd_system import CmdSystem
from hybrid_system import HybridSystem, add_hybrid_system_arguments

# from ranker_system import IRRankerSystem, NeuralRankerSystem
# from model.retriever import Retriever, add_retriever_arguments
from model.generator import Templates, Generator
from model.manager import Manager

def add_system_arguments(parser):
    # NOTE: hybrid system arguments are covered by neural system and rulebased system
    #add_hybrid_system_arguments(parser)
    add_price_tracker_arguments(parser)
    add_neural_system_arguments(parser)
    add_rulebased_arguments(parser)
    # add_retriever_arguments(parser)
    # add_configurable_rulebased_arguments(parser)

def get_system(name, args, schema=None, timed=False, model_path=None):
    lexicon = PriceTracker(args.price_tracker_model)
    if name == 'rulebased':
        templates = Templates.from_pickle(args.templates)
        generator = Generator(templates)
        manager = Manager.from_pickle(args.policy)
        return RulebasedSystem(lexicon, generator, manager, timed)
    elif name == 'hybrid':
        templates = Templates.from_pickle(args.templates)
        manager = PytorchNeuralSystem(args, schema, lexicon, model_path, timed)
        generator = Generator(templates)
        return HybridSystem(lexicon, generator, manager, timed)
    elif name == 'cmd':
        return CmdSystem()
    elif name == 'pt-neural':
        assert model_path
        return PytorchNeuralSystem(args, schema, lexicon, model_path, timed)
    else:
        raise ValueError('Unknown system %s' % name)
