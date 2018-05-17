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

parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
parser.add_argument('--agents', help='What kind of agent to use', nargs='*')
parser.add_argument('--scenario-offset', default=0, type=int,
        help='Number of scenarios to skip at the beginning')
parser.add_argument('--remove-fail', default=False, action='store_true',
        help='Remove failed dialogues')
parser.add_argument('--max-turns', default=100, type=int,
        help='Maximum number of turns')
parser.add_argument('--results-path', default=None,
        help='json path to store the results of the chat examples')
parser.add_argument('--max-examples', default=20, type=int,
        help='Number of test examples to predict')
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

if not args.agents:
    args.agents = ['rulebased', 'rulebased']
model_path = args.checkpoint
agents = [get_system(name, args, schema, model_path=model_path) for name in args.agents]

global num_examples
num_examples = args.scenario_offset

summary_map = {}

if args.train_max_examples or args.test_max_examples:
    raise ValueError("Historically, this file was for generating training data, \
        but is now used for verifying trained models.  Thus, we no longer use \
        train_max_examples or test_max_examples. Please fill in results_path to \
        specify where you want to store results and max_examples to specify how many.")

examples = []
num_failed = 0
scenarios = scenario_db.scenarios_list
#scenarios = [scenario_db.scenarios_map['S_8COuPdjZZkYgrzhb']]
random.shuffle(scenarios)
for i in range(args.max_examples):
    scenario = scenarios[num_examples % len(scenario_db.scenarios_list)]
    sessions = [agents[0].new_session(0, scenario.kbs[0]), agents[1].new_session(1, scenario.kbs[1])]
    controller = Controller(scenario, sessions)
    ex = controller.simulate(args.max_turns, verbose=args.verbose)
    if not controller.complete():
        num_failed += 1
        if args.remove_fail:
            continue
    examples.append(ex)
    num_examples += 1
if args.results_path is not None:
    with open(args.results_path, 'w') as out:
        print >>out, json.dumps([e.to_dict() for e in examples])
if num_failed == 0:
    print 'All {} dialogues succeeded!'.format(num_examples)
else:
    print 'Number of failed dialogues:', num_failed


