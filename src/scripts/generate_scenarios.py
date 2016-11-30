#!/usr/bin/env python

import argparse
import random
import numpy as np
from itertools import izip
from src.basic.util import random_multinomial, generate_uuid, write_json
from src.basic.schema import Schema
from src.basic.scenario_db import Scenario, ScenarioDB, add_scenario_arguments
from src.basic.kb import KB

parser = argparse.ArgumentParser()
parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
parser.add_argument('--num-scenarios', help='Number of scenarios to generate', type=int, default=1)
parser.add_argument('--num-items', help='Number of items to generate per scenario', type=int, default=10)
parser.add_argument('--domain', help='{MutualFriends, Matchmaking}', default=None)
parser.add_argument('--random-attributes', action='store_true',
                    help='If specified, uses a random number and subset of attributes for each scenario')
add_scenario_arguments(parser)
args = parser.parse_args()
if args.random_seed:
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)


def get_multinomial(alpha, n):
    return np.random.dirichlet([alpha] * n)


def generate_scenario(schema):
    num_items = args.num_items
    alphas = schema.alphas
    random_attributes = args.random_attributes
    scenario_attributes = schema.attributes
    if random_attributes:
        # sample random number and set of attributes
        num_attributes = min(np.random.choice(xrange(3, 5)), len(schema.attributes))
        scenario_attributes = np.random.choice(schema.attributes, num_attributes, replace=False)
        scenario_attributes = schema.get_ordered_attribute_subset(scenario_attributes)
        alphas = dict((attr, schema.alphas[attr]) for attr in scenario_attributes)

    # Generate the profile of the two agents
    agents = (0, 1)
    distribs = ({}, {})
    values = {}  # {attr_name: possible values}
    num_values = num_items * 2
    for attr, alpha in alphas.iteritems():
        n = min(len(schema.values[attr.value_type]), num_values)
        values[attr.name] = random.sample(schema.values[attr.value_type], n)
        for agent in agents:
            distribs[agent][attr.name] = get_multinomial(alpha, n)

    # Generate items for each agent until we get a match
    agent_item_hashes = ({}, {})
    agent_items = ([], [])
    match = None
    for i in range(100000):
        if match and len(agent_items[0]) >= num_items and len(agent_items[1]) >= num_items:
            break

        for agent in agents:
            # Create the item (and its hash)
            item = {}
            item_hash = []
            for attr in scenario_attributes:
                distrib = distribs[agent][attr.name]
                index = random_multinomial(distrib)
                # print attr.name, index, len(values[attr.name])
                value = values[attr.name][index]
                item[attr.name] = value
                item_hash.append(index)
            item_hash = ' '.join(map(str, item_hash))

            # Make sure no duplicated entries
            if item_hash in agent_item_hashes[agent]:
                continue

            if item_hash in agent_item_hashes[1 - agent]:
                # Make sure there's exactly one match
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
    kbs = [KB(scenario_attributes, items) for items in agent_items]
    scenario = Scenario(generate_uuid('S'), scenario_attributes, kbs)
    return scenario

# Generate scenarios
schema = Schema(args.schema_path, args.domain)
scenario_db = ScenarioDB([generate_scenario(schema) for i in range(args.num_scenarios)])
write_json(scenario_db.to_dict(), args.scenarios_path[0])

# Output a sample of what we've generated
for i in range(min(100, len(scenario_db.scenarios_list))):
    print ''
    for agent in (0, 1):
        kb = scenario_db.scenarios_list[i].kbs[agent]
        kb.dump()
