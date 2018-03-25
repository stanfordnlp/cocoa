'''
Takes two agent implementations and generates the dialogues.
'''

import argparse
import random
import json
import numpy as np

from cocoa.core.util import read_json
from cocoa.core.schema import Schema
from cocoa.core.scenario_db import ScenarioDB, add_scenario_arguments
from cocoa.core.dataset import add_dataset_arguments

from core.scenario import Scenario
from core.controller import Controller
from systems import add_system_arguments, get_system

parser = argparse.ArgumentParser()
parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
parser.add_argument('--agents', help='What kind of agent to use', nargs='*')
parser.add_argument('--scenario-offset', default=0, type=int, help='Number of scenarios to skip at the beginning')
parser.add_argument('--remove-fail', default=False, action='store_true', help='Remove failed dialogues')
parser.add_argument('--max-turns', default=100, type=int, help='Maximum number of turns')
parser.add_argument('-v', '--verbose', default=False, action='store_true', help='whether or not to have verbose prints')
add_scenario_arguments(parser)
add_dataset_arguments(parser)
add_system_arguments(parser)
args = parser.parse_args()
if args.random_seed:
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

schema = Schema(args.schema_path)
scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path), Scenario)

if args.train_max_examples is None:
    args.train_max_examples = scenario_db.size
if args.test_max_examples is None:
    args.test_max_examples = scenario_db.size

if not args.agents:
    args.agents = ['rulebased', 'rulebased']
if hasattr(args, 'checkpoint'):
    model_path = args.checkpoint
else:
    model_path = None
agents = [get_system(name, args, schema, model_path=model_path) for name in args.agents]
num_examples = args.scenario_offset

summary_map = {}
def generate_examples(description, examples_path, max_examples, remove_fail, max_turns):
    global num_examples
    examples = []
    num_failed = 0
    scenarios = scenario_db.scenarios_list
    #scenarios = [scenario_db.scenarios_map['S_8COuPdjZZkYgrzhb']]
    random.shuffle(scenarios)
    for i in range(max_examples):
        scenario = scenarios[num_examples % len(scenario_db.scenarios_list)]
        sessions = [agents[0].new_session(0, scenario.kbs[0]), agents[1].new_session(1, scenario.kbs[1])]
        controller = Controller(scenario, sessions)
        ex = controller.simulate(max_turns, verbose=args.verbose)
        if not controller.complete():
            num_failed += 1
            if remove_fail:
                continue
        examples.append(ex)
        num_examples += 1
    with open(examples_path, 'w') as out:
        print >>out, json.dumps([e.to_dict() for e in examples])
    if num_failed == 0:
        print 'All {} dialogues succeeded!'.format(num_examples)
    else:
        print 'Number of failed dialogues:', num_failed

if args.train_max_examples:
    generate_examples('train', args.train_examples_paths[0], args.train_max_examples, args.remove_fail, args.max_turns)
if args.test_max_examples:
    generate_examples('test', args.test_examples_paths[0], args.test_max_examples, args.remove_fail, args.max_turns)
