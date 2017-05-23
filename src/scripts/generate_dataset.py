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
from src.basic.systems.simple_system import SimpleSystem
from src.basic.systems.cmd_system import CmdSystem
from src.basic.systems.neural_system import NeuralSystem, add_neural_system_arguments
from src.basic.controller import Controller
from src.basic.lexicon import Lexicon, add_lexicon_arguments
from src.basic.inverse_lexicon import InverseLexicon
from src.lib import logstats
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
parser.add_argument('--agents', help='What kind of agent to use {heuristic}', nargs='*')
parser.add_argument('--model-path', help='Path to model (used for neural agents)')
parser.add_argument('--scenario-offset', default=0, type=int, help='Number of scenarios to skip at the beginning')
parser.add_argument('--remove-fail', default=False, action='store_true', help='Remove failed dialogues')
parser.add_argument('--stats-file', default='stats.json', help='Path to save json statistics (dataset, training etc.) file')
parser.add_argument('--max-turns', default=100, type=int, help='Maximum number of turns')
add_scenario_arguments(parser)
add_lexicon_arguments(parser)
add_dataset_arguments(parser)
add_neural_system_arguments(parser)
args = parser.parse_args()
logstats.init(args.stats_file)
if args.random_seed:
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

schema = Schema(args.schema_path)
scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))
lexicon = Lexicon(schema, args.learned_lex, stop_words=args.stop_words)
if args.inverse_lexicon:
    realizer = InverseLexicon(schema, args.inverse_lexicon)
else:
    realizer = None

if args.train_max_examples is None:
    args.train_max_examples = scenario_db.size
if args.test_max_examples is None:
    args.test_max_examples = scenario_db.size

def get_system(name):
    if name == 'simple':
        return SimpleSystem(lexicon, realizer=realizer)
    elif name == 'neural':
        assert args.model_path
        return NeuralSystem(schema, lexicon, args.model_path, args.decoding, realizer=realizer)
    elif name == 'cmd':
        return CmdSystem()
    else:
        raise ValueError('Unknown system %s' % name)

if not args.agents:
    args.agents = ['simple', 'simple']
agents = [get_system(name) for name in args.agents]
num_examples = args.scenario_offset

summary_map = {}
def generate_examples(description, examples_path, max_examples, remove_fail, max_turns):
    global num_examples
    examples = []
    num_failed = 0
    for i in range(max_examples):
        scenario = scenario_db.scenarios_list[num_examples % len(scenario_db.scenarios_list)]
        sessions = [agents[0].new_session(0, scenario.kbs[0]), agents[1].new_session(1, scenario.kbs[1])]
        controller = Controller(scenario, sessions)
        ex = controller.simulate(max_turns)
        if ex.outcome['reward'] == 0:
            num_failed += 1
            if remove_fail:
                continue
        examples.append(ex)
        num_examples += 1
        logstats.update_summary_map(summary_map, {'length': len(ex.events)})
    with open(examples_path, 'w') as out:
        print >>out, json.dumps([e.to_dict() for e in examples])
    print 'number of failed dialogues:', num_failed

    logstats.add('length', summary_map['length']['mean'])

if args.train_max_examples:
    generate_examples('train', args.train_examples_paths[0], args.train_max_examples, args.remove_fail, args.max_turns)
if args.test_max_examples:
    generate_examples('test', args.test_examples_paths[0], args.test_max_examples, args.remove_fail, args.max_turns)
