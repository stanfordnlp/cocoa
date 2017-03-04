#!/usr/bin/env python

import argparse
import random
import numpy as np
from src.basic.schema import Schema
from src.basic.scenario_db import NegotiationScenario, ScenarioDB, add_scenario_arguments
from src.basic.util import generate_uuid, write_json
from src.basic.kb import KB

def generate_kb(schema):
    item = {}
    for attr in schema.attributes:
        value_set = schema.values[attr.value_type]
        if attr.multivalued:
            num_values = np.random.randint(1, 4)  # 1 to 3 values
            value = tuple(np.random.choice(value_set, num_values, replace=False))
        else:
            value = random.choice(value_set)
        item[attr.name] = value
    return KB(schema.attributes, [item])

def generate_targets(base, intersections):
    '''
    base: seller's bottom line
    intersections: a set of possible intersections with the buyer's best price
    '''
    diff = random.choice(intersections)
    return {'seller': base,
            'buyer': base + diff,
            }

def generate_scenario(schema, base, intersections):
    kb = generate_kb(schema)
    targets = generate_targets(base, intersections)
    return NegotiationScenario(generate_uuid('S'), schema.attributes, [kb, kb], targets)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    parser.add_argument('--num-scenarios', help='Number of scenarios to generate', type=int, default=1)
    parser.add_argument('--intersections', nargs='*', type=float, default=[100, 200, 300], help="Intersection of buyer and seller's price range")
    add_scenario_arguments(parser)
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    schema = Schema(args.schema_path)

    scenario_list = []
    for i in xrange(args.num_scenarios):
        s = generate_scenario(schema, 500, args.intersections)
        scenario_list.append(s)
    scenario_db = ScenarioDB(scenario_list)
    write_json(scenario_db.to_dict(), args.scenarios_path)

    for i in range(min(100, len(scenario_db.scenarios_list))):
        print '---------------------------------------------------------------------------------------------'
        print '---------------------------------------------------------------------------------------------'
        scenario = scenario_db.scenarios_list[i]
        print "Scenario id: %s" % scenario.uuid
        print "Targets: %s" % scenario.targets
        for agent in (0, 1):
            kb = scenario.kbs[agent]
            kb.dump()
