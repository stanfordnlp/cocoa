from src.basic.negotiation.price_tracker import PriceTracker
from src.model.negotiation.retriever import Retriever, add_retriever_arguments
from rulebased_system import RulebasedSystem
from ranker_system import IRRankerSystem, EncDecRankerSystem
#from neural_system import NeuralSystem, add_neural_system_arguments
from cmd_system import CmdSystem

def add_system_arguments(parser):
    #add_neural_system_arguments(parser)
    add_retriever_arguments(parser)
    parser.add_argument('--mappings', default='.', help='Directory to save mappings/vocab')

def get_system(name, args, schema=None, timed=False):
    lexicon = PriceTracker()
    if name == 'rulebased':
        return RulebasedSystem(lexicon, timed)
    elif name == 'cmd':
        return CmdSystem()
    elif name.startswith('ranker'):
        # TODO: hack
        #retriever1 = Retriever(args.index+'-1', context_size=args.retriever_context_len, num_candidates=args.num_candidates)
        #retriever2 = Retriever(args.index+'-2', context_size=args.retriever_context_len, num_candidates=args.num_candidates)
        retriever = Retriever(args.index, context_size=args.retriever_context_len, num_candidates=args.num_candidates)
        if name == 'ranker-ir':
            return IRRankerSystem(schema, lexicon, retriever)
        elif name == 'ranker-ir1':
            return IRRankerSystem(schema, lexicon, retriever1)
        elif name == 'ranker-ir2':
            return IRRankerSystem(schema, lexicon, retriever2)
        elif name == 'ranker-encdec':
            return EncDecRankerSystem(schema, lexicon, retriever, args.model_path, args.mappings)
        else:
            raise ValueError
    #elif name == 'neural':
    #    assert args.model_path
    #    return NeuralSystem(schema, lexicon, args.model_path, args.decoding)
    else:
        raise ValueError('Unknown system %s' % name)
