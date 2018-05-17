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

def generate_examples(agents, agent_names, scenarios, num_examples, max_turns):
    examples = []
    for i in range(num_examples):
        scenario = scenarios[i % len(scenarios)]
        # Each agent needs to play both buyer and seller
        for j in (0, 1):
            new_agents = [agents[j], agents[1-j]]
            new_agent_names = [agent_names[j], agent_names[1-j]]
            sessions = [new_agents[0].new_session(0, scenario.kbs[0]),
                    new_agents[1].new_session(1, scenario.kbs[1])]
            controller = Controller(scenario, sessions, session_names=new_agent_names)
            ex = controller.simulate(max_turns, verbose=args.verbose)
            examples.append(ex)
    return examples

if __name__ == '__main__':
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--agent', nargs=3, metavar=('type', 'checkpoint', 'name'), action='append', help='Agent parameters')
    parser.add_argument('--max-turns', default=20, type=int, help='Maximum number of turns')
    parser.add_argument('--num-examples', type=int)
    parser.add_argument('--examples-path')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='whether or not to have verbose prints')
    add_scenario_arguments(parser)
    add_system_arguments(parser)
    args = parser.parse_args()

    schema = Schema(args.schema_path)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path), Scenario)

    agents = {}
    for agent_params in args.agent:
        agent_type, model_path, agent_name = agent_params
        agents[agent_name] = get_system(agent_type, args, schema, model_path=model_path)

    scenarios = scenario_db.scenarios_list
    examples = []
    for base_agent_name in ('sl-words',):
        base_agent = agents[base_agent_name]
        for agent_name, agent in agents.iteritems():
            if agent_name != base_agent_name:
                agents = [base_agent, agent]
                agent_names = [base_agent_name, agent_name]
                examples.extend(generate_examples(agents, agent_names, scenarios, args.num_examples, args.max_turns))

    with open(args.examples_path, 'w') as out:
        print >>out, json.dumps([e.to_dict() for e in examples])

