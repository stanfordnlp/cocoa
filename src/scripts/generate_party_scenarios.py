#!/usr/bin/env python

import argparse
import random
import numpy as np
import copy
from itertools import izip
from src.basic.schema import Schema
from src.basic.scenario_db import PartyScenario, ScenarioDB, add_scenario_arguments
from src.basic.util import generate_uuid, write_json
from src.basic.kb import PartyKB

def generate_kbs(schema):
    party_params = {}
    for attr in schema.attributes:
        value_set = schema.values[attr.value_type]
        if attr.multivalued:
            num_values = np.random.randint(1, 4)  # 1 to 3 values
            value = tuple(np.random.choice(value_set, num_values, replace=False))
        else:
            value = random.choice(value_set)
        party_params[attr.name] = value
    party_kb = PartyKB(schema.attributes, party_params)
    kbs = [party_kb]
    return kbs

def generate_scenario(schema):
    kbs = generate_kbs(schema)
    return PartyScenario(generate_uuid('S'), schema.attributes, kbs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    parser.add_argument('--num-scenarios', help='Number of scenarios to generate', type=int, default=1)
    add_scenario_arguments(parser)
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    schema = Schema(args.schema_path)

    scenario_list = []
    for i in xrange(args.num_scenarios):
        s = generate_scenario(schema)
        scenario_list.append(s)
    scenario_db = ScenarioDB(scenario_list)
    write_json(scenario_db.to_dict(), args.scenarios_path)

    for i in range(min(100, len(scenario_db.scenarios_list))):
        print '---------------------------------------------------------------------------------------------'
        print '---------------------------------------------------------------------------------------------'
        scenario = scenario_db.scenarios_list[i]
        print "Scenario id: %s" % scenario.uuid
        kb = scenario.kbs[0]
        kb.dump()
