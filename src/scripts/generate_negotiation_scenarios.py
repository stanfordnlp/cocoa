#!/usr/bin/env python

import argparse
import random
import numpy as np
import copy
from itertools import izip
from src.basic.schema import Schema
from src.basic.scenario_db import NegotiationScenario, ScenarioDB, add_scenario_arguments
from src.basic.util import generate_uuid, write_json
from src.basic.kb import NegotiationKB

private_attr = ['Laundry', 'Pet', 'Built data', 'Neighborhood']

def generate_kbs(schema):
    buyer_item, seller_item = {}, {}
    for attr in schema.attributes:
        if attr.name in ('Role', 'Target', 'Bottomline'):
            continue
        value_set = schema.values[attr.value_type]
        if attr.multivalued:
            num_values = np.random.randint(1, 4)  # 1 to 3 values
            value = tuple(np.random.choice(value_set, num_values, replace=False))
        else:
            value = random.choice(value_set)
        seller_item[attr.name] = value
        if attr.name in private_attr and random.random() < 0.5:
            buyer_item[attr.name] = None
        else:
            buyer_item[attr.name] = value
    seller_kb = NegotiationKB(schema.attributes, [seller_item])
    buyer_kb = NegotiationKB(schema.attributes, [buyer_item])
    kbs = [None, None]
    kbs[NegotiationScenario.BUYER] = buyer_kb
    kbs[NegotiationScenario.SELLER] = seller_kb
    return kbs

def generate_price_range(base, intersections):
    '''
    base: seller's bottom line
    intersections: a set of possible intersections with the buyer's best price
    '''
    diff = random.choice(intersections)
    seller_bottomline = base
    buyer_bottomline = base + diff
    seller_target = buyer_bottomline + 100
    buyer_target = seller_bottomline - 100
    return {'seller': {'Bottomline': seller_bottomline, 'Target': seller_target},
            'buyer': {'Bottomline': buyer_bottomline, 'Target': buyer_target}
            }

def generate_scenario(schema, base, intersections):
    kbs = generate_kbs(schema)
    ranges = generate_price_range(base, intersections)
    for i, (role, price) in enumerate(ranges.iteritems()):
        item = kbs[i].items[0]
        item['Role'] = role
        for k, v in price.iteritems():
            item[k] = v
    return NegotiationScenario(generate_uuid('S'), schema.attributes, kbs)

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
        for agent in (0, 1):
            kb = scenario.kbs[agent]
            kb.dump()
