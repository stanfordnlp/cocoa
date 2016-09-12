#!/usr/bin/env python

import argparse
import random, sys
import numpy as np
from itertools import izip
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
    np.random.seed(args.random_seed)

def get_multinomial(alpha, n):
    return np.random.dirichlet([alpha] * n)

def generate_scenario(schema):
    '''
    Generates scenarios
    '''
    num_items = args.num_items
    alphas = list(np.linspace(2, 0.3, len(schema.attributes)-1))
    np.random.shuffle(alphas)
    # The first attribute (Name) always have a near-uniform distribution
    alphas = [3] + list(alphas)

    # Generate the profile of the two agents
    agents = (0, 1)
    distribs = ([], [])
    for agent in agents:
        for attr, alpha in izip(schema.attributes, alphas):
            values = schema.values[attr.value_type]
            distribs[agent].append(get_multinomial(alpha, len(values)))

    # Generate items for each agent until we get a match
    agent_item_hashes = ({}, {})
    agent_items = ([], [])
    match = None
    # TODO: make sure exactly one match
    for i in range(100000):
        if match and len(agent_items[0]) >= num_items and len(agent_items[1]) >= num_items:
            break

        for agent in agents:
            # Create the item (and its hash)
            item = {}
            item_hash = []
            for attr, distrib in zip(schema.attributes, distribs[agent]):
                index = random_multinomial(distrib)
                value = schema.values[attr.value_type][index]
                item[attr.name] = value
                item_hash.append(index)
            item_hash = ' '.join(map(str, item_hash))

            # Make sure no duplicated entries
            if item_hash in agent_item_hashes[agent]:
                continue

            if item_hash in agent_item_hashes[1 - agent]:
                # Make sure the match is unique
                if match is not None:
                    continue
                match = [None, None]
                match[agent] = len(agent_items[agent])
                match[1 - agent] = agent_item_hashes[1 - agent][item_hash]

            # Add the item
            agent_item_hashes[agent][item_hash] = len(agent_items[agent])
            agent_items[agent].append(item)


    if not match:
        raise Exception('Failed to match')

    # Move the matching item to the top
    def swap(l, i, j):
        t = l[i]
        l[i] = l[j]
        l[j] = t
    for agent in agents:
        swap(agent_items[agent], 0, match[agent])

    # Truncate to num_items
    for agent in agents:
        del agent_items[agent][num_items:]

    # Truncate to num_items and enforce uniqueness
    #new_agent_items = ([], [])
    #for agent in agents:
    #    unique_attrs = {attr.name: set() for attr in schema.attributes if attr.unique}
    #    for item in agent_items[agent]:
    #        unique = True
    #        for attr, values in unique_attrs.iteritems():
    #            if item[attr] in values:
    #                unique = False
    #                break
    #        if not unique:
    #            continue
    #        else:
    #            for attr, values in unique_attrs.iteritems():
    #                values.add(item[attr])
    #            new_agent_items[agent].append(item)
    #            if len(new_agent_items[agent]) == num_items:
    #                break
    #agent_items = new_agent_items
    for agent in agents:
        if len(agent_items[agent]) != num_items:
            raise Exception('Failed to generate enough items')

    # Shuffle items
    for agent in agents:
        np.random.shuffle(agent_items[agent])

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
#for i in range(args.num_scenarios):
#    scenarios = generate_scenario(schema)
#    scenarios.kbs[0].dump()
#    scenarios.kbs[1].dump()
#sys.exit()

scenario_db = ScenarioDB([generate_scenario(schema) for i in range(args.num_scenarios)])
write_json(scenario_db.to_dict(), args.scenarios_path)

# Output a sample of what we've generated
for agent in (0, 1):
    kb = scenario_db.scenarios_list[0].kbs[agent]
    kb.dump()
