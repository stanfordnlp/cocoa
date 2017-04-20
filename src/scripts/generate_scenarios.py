#!/usr/bin/env python

import argparse
import random
import numpy as np
from itertools import izip
import sys
from src.basic.util import random_multinomial, generate_uuid, write_json
from src.basic.schema import Schema
from src.basic.scenario_db import Scenario, ScenarioDB, add_scenario_arguments
from src.basic.kb import KB

parser = argparse.ArgumentParser()
parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
parser.add_argument('--num-scenarios', help='Number of scenarios to generate', type=int, default=1)
parser.add_argument('--num-items', help='Number of items to generate per scenario', type=int, default=10)
parser.add_argument('--domain', help='{MutualFriends, Matchmaking}', default=None)


def add_randomization_arguments(parser):
    parser.add_argument('--random-attributes', action='store_true',
                        help='If specified, uses a random number, distribution, and subset of attributes for each '
                             'scenario')
    parser.add_argument('--min-attributes', type=int, default=3, help='Minimum number of attributes per scenario')
    parser.add_argument('--max-attributes', type=int, default=4, help='Maximum number of attributes per scenario')

    parser.add_argument('--random-items', action='store_true',
                        help='If specified, selects a random number of items (by default in the range [5,10]) for each scenario.')
    parser.add_argument('--min-items', type=int, default=5,
                        help='Minimum number of items per scenario')
    parser.add_argument('--max-items', type=int, default=12,
                        help='Minimum number of items per scenario')

    parser.add_argument('--alphas', nargs='*', type=float, default=[0.3, 1.0, 3.0],
                        help='Alpha values to select from for each attribute.')

add_scenario_arguments(parser)
add_randomization_arguments(parser)
args = parser.parse_args()
if args.random_seed:
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)


def get_multinomial(alpha, n):
    return np.random.dirichlet([alpha] * n)


def select_alphas(attributes):
    alphas = np.random.choice(args.alphas, size=len(attributes))
    return dict(zip(attributes, alphas))


def generate_scenario(schema):
    num_items = args.num_items
    if args.random_items:
        num_items = np.random.choice(xrange(args.min_items, args.max_items+1))
    alphas = schema.alphas
    random_attributes = args.random_attributes
    scenario_attributes = schema.attributes
    if random_attributes:
        # sample random number and set of attributes, and choose alphas for each attribute
        num_attributes = min(np.random.choice(xrange(args.min_attributes, args.max_attributes)), len(schema.attributes))
        scenario_attributes = np.random.choice(schema.attributes, num_attributes, replace=False)
        scenario_attributes = schema.get_ordered_attribute_subset(scenario_attributes)
        alphas = select_alphas(scenario_attributes)

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
        print >> sys.stderr,  'Failed to match'
        return None

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
    scenario = Scenario(generate_uuid('S'), scenario_attributes, kbs, [alphas[attr] for attr in scenario_attributes])
    return scenario

# Generate scenarios
schema = Schema(args.schema_path, args.domain)
scenario_list = []
while len(scenario_list) < args.num_scenarios:
    s = generate_scenario(schema)
    if s is not None:
        scenario_list.append(s)

scenario_db = ScenarioDB(scenario_list)
write_json(scenario_db.to_dict(), args.scenarios_path)

# Output a sample of what we've generated
for i in range(min(100, len(scenario_db.scenarios_list))):
    print '---------------------------------------------------------------------------------------------'
    print '---------------------------------------------------------------------------------------------'
    scenario = scenario_db.scenarios_list[i]
    print "Scenario id: %s" % scenario.uuid
    print "Alphas: [%s]" % ", ".join(["%2.1f" % alpha for alpha in scenario.alphas])
    for agent in (0, 1):

        kb = scenario.kbs[agent]
        kb.dump()
