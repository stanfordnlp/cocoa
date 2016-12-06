'''
Takes two agent implementations and generates the dialogues.
'''

import argparse
import random
import json
from src.basic.util import read_json
from src.basic.schema import Schema
from src.basic.scenario_db import ScenarioDB, add_scenario_arguments
from src.basic.dataset import add_dataset_arguments
from src.basic.systems.heuristic_system import HeuristicSystem, add_heuristic_system_arguments
from src.basic.systems.simple_system import SimpleSystem
from src.basic.systems.neural_system import NeuralSystem
from src.basic.controller import Controller
from src.basic.lexicon import Lexicon
from src.lib import logstats

parser = argparse.ArgumentParser()
parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
parser.add_argument('--agents', help='What kind of agent to use {heuristic}', nargs='*')
parser.add_argument('--model-path', help='Path to model (used for neural agents)')
parser.add_argument('--scenario-offset', default=0, type=int, help='Number of scenarios to skip at the beginning')
parser.add_argument('--remove-fail', default=False, action='store_true', help='Remove failed dialogues')
parser.add_argument('--stats-file', default='stats.json', help='Path to save json statistics (dataset, training etc.) file')
parser.add_argument('--fact-check', default=False, action='store_true', help='Check if the utterance is true given the KB. Only work for simulated data.')
add_scenario_arguments(parser)
add_dataset_arguments(parser)
add_heuristic_system_arguments(parser)
args = parser.parse_args()
logstats.init(args.stats_file)
if args.random_seed:
    random.seed(args.random_seed)

schema = Schema(args.schema_path)
scenario_db = ScenarioDB.from_dict(schema, (read_json(path) for path in args.scenarios_path))
lexicon = Lexicon(schema, learned_lex=False)

def get_system(name):
    if name == 'simple':
        return SimpleSystem()
    elif name == 'heuristic':
        return HeuristicSystem(args.joint_facts, args.ask)
    elif name == 'neural':
        assert args.model_path
        return NeuralSystem(schema, lexicon, args.model_path)
    else:
        raise ValueError('Unknown system %s' % name)

if not args.agents:
    args.agents = ['simple', 'simple']
agents = [get_system(name) for name in args.agents]
num_examples = args.scenario_offset

summary_map = {}
def generate_examples(description, examples_path, max_examples, remove_fail):
    global num_examples
    examples = []
    num_failed = 0
    for i in range(max_examples):
        scenario = scenario_db.scenarios_list[num_examples % len(scenario_db.scenarios_list)]
        sessions = [agents[0].new_session(0, scenario.kbs[0]), agents[1].new_session(1, scenario.kbs[1])]
        controller = Controller(scenario, sessions)
        ex = controller.simulate()
        if remove_fail and ex.outcome['reward'] == 0:
            num_failed += 1
            continue
        examples.append(ex)
        num_examples += 1
        logstats.update_summary_map(summary_map, {'length': len(ex.events)})
    with open(examples_path, 'w') as out:
        print >>out, json.dumps([e.to_dict() for e in examples])
    print 'number of failed dialogues:', num_failed

    logstats.add('length', summary_map['length']['mean'])
    if args.fact_check:
        if args.agents[0] == args.agents[1] and hasattr(agents[0], 'env'):
            results0 = agents[0].env.evaluator.report()
            results1 = agents[1].env.evaluator.report()
            results = {k: (results0[k] + results1[k]) / 2. for k in results0}
            logstats.add('bot_chat', results)

if args.train_max_examples:
    generate_examples('train', args.train_examples_paths[0], args.train_max_examples, args.remove_fail)
if args.test_max_examples:
    generate_examples('test', args.test_examples_paths[0], args.test_max_examples, args.remove_fail)
