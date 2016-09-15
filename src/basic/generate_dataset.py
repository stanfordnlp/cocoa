'''
Takes two agent implementations and generates the dialogues.
'''

import argparse
import random
from util import read_json
from schema import Schema
from scenario_db import ScenarioDB, add_scenario_arguments
from dataset import add_dataset_arguments
from simple_system import SimpleSystem
from exact_system import ExactSystem
from heuristic_system import HeuristicSystem
from controller import Controller

parser = argparse.ArgumentParser()
parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
parser.add_argument('--agents', help='What kind of agent to use {simple, exact, heuristic}', nargs='*')
add_scenario_arguments(parser)
add_dataset_arguments(parser)
args = parser.parse_args()
if args.random_seed:
    random.seed(args.random_seed)

schema = Schema(args.schema_path)
scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))

agent_systems = {'simple': SimpleSystem, 'exact': ExactSystem, 'heuristic': HeuristicSystem}
if not args.agents:
    args.agents = ['simple', 'simple']
agents = [agent_systems[name] for name in args.agents]

def generate_examples(description, examples_path, max_examples):
    examples = []
    for i in range(max_examples):
        scenario = scenario_db.scenarios_list[i % len(scenario_db.scenarios_list)]
        systems = [agents[0](0, scenario.kbs[0]), agents[1](1, scenario.kbs[1])]
        controller = Controller(scenario, systems)
        ex = controller.run()
        examples.append(ex)
        break
    #with open(examples_path, 'w') as out:
    #    print >>out, json.dumps([e.to_dict() for e in examples])

if args.train_max_examples:
    generate_examples('train', args.train_examples_paths[0], args.train_max_examples)
if args.test_max_examples:
    generate_examples('test', args.test_examples_paths[0], args.test_max_examples)
