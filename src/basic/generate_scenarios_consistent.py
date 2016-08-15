#!/usr/bin/env python

import argparse
import random
import numpy
from util import random_multinomial, generate_uuid, write_json
from schema import Schema
from scenario_db import Scenario, ScenarioDB, add_scenario_arguments
from kb import KB

parser = argparse.ArgumentParser()
parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
parser.add_argument('--num-scenarios', help='Number of scenarios to generate', type=int, default=1)
parser.add_argument('--num-items', help='Number of items to generate per scenario', type=int, default=10)
parser.add_argument('--alpha', help='Concentration parameter (large means more uniform)', type=float, default=1)
add_scenario_arguments(parser)
args = parser.parse_args()
if args.random_seed:
    random.seed(args.random_seed)
    numpy.random.seed(args.random_seed)

alpha = args.alpha

def generate_scenario(schema):
    '''
    Generates scenarios
    '''
    num_items = args.num_items

    # Generate the profile of the two agents
    agents = (0, 1)
    agent_items = []
    N = len(people)
    mutual_person_index = numpy.random.randint(N)
    inds = range(mutual_person_index+1) + range(mutual_person_index+1, N)
    for agent in agents:
        random.shuffle(inds)
        agent_item = [people[mutual_person_index]] + [people[ind] for ind in inds[:num_items-1]]
        agent_items.append(agent_item)

    # Shuffle items
    for agent in agents:
        numpy.random.shuffle(agent_items[agent])

    # Check that there is at least one match
    matches = []
    for i in range(num_items):
        for j in range(num_items):
            if agent_items[0][i] == agent_items[1][j]:
                matches.append((i, j))
    if len(matches) == 0:
        raise Exception('Internal error: expected at least one match, but got: %s' % matches)

    # Create the scenario
    kbs = [KB(schema, items) for items in agent_items]
    scenario = Scenario(generate_uuid('S'), kbs)
    return scenario

# Generate scenarios
schema = Schema(args.schema_path)

# Generate a list of people that are used consistently in all scenarios
people = []
for name in schema.values['person']:
    item = {}
    for attr in schema.attributes:
        values = schema.values[attr.value_type]
        distrib = numpy.random.dirichlet([alpha] * len(values))
        index = random_multinomial(distrib)
        value = schema.values[attr.value_type][index]
        item[attr.name] = value
    item['Name'] = name
    people.append(item)

scenario_db = ScenarioDB([generate_scenario(schema) for i in range(args.num_scenarios)])
write_json(scenario_db.to_dict(), args.scenarios_path)

# Output a sample of what we've generated
for agent in (0, 1):
    kb = scenario_db.scenarios_list[0].kbs[agent]
    kb.dump()
