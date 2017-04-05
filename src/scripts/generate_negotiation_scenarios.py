#!/usr/bin/env python

import argparse
import random
import numpy as np
import copy
from itertools import izip
from src.basic.schema import Schema
from src.basic.scenario_db import NegotiationScenario, ScenarioDB, add_scenario_arguments
from src.basic.util import generate_uuid, write_json, read_json
from src.basic.kb import NegotiationKB

private_attr = ['Laundry', 'Pet', 'Built data', 'Neighborhood']
BUYER = NegotiationScenario.BUYER
SELLER = NegotiationScenario.SELLER

def generate_simulated_kbs(schema):
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
    seller_kb = NegotiationKB(schema.attributes, {'personal': {'Role': 'seller'}, 'item': seller_item})
    buyer_kb = NegotiationKB(schema.attributes, {'personal': {'Role': 'buyer'}, 'item': buyer_item})
    kbs = [None, None]
    kbs[BUYER] = buyer_kb
    kbs[SELLER] = seller_kb
    return kbs

def generate_scraped_kbs(schema, listing):
    buyer_item, seller_item = {}, {}
    for attr in schema.attributes:
        if attr.name in ('Role', 'Target', 'Bottomline'):
            continue
        buyer_item[attr.name] = listing[attr.name]
        seller_item[attr.name] = listing[attr.name]
    seller_kb = NegotiationKB(schema.attributes, {'personal': {'Role': 'seller'}, 'item': seller_item})
    buyer_kb = NegotiationKB(schema.attributes, {'personal': {'Role': 'buyer'}, 'item': buyer_item})
    kbs = [None, None]
    kbs[BUYER] = buyer_kb
    kbs[SELLER] = seller_kb
    return kbs

def generate_kbs(schema, listing):
    if not listing:
        return generate_simulated_kbs(schema)
    else:
        return generate_scraped_kbs(schema, listing)

def discretize(price, price_unit):
    price = int(price / price_unit)
    return price

def generate_price_range(base_price, price_unit, intersections, flexibility=0.2):
    '''
    base: a middle point to generate the range
    intersections: percentage of intersection relative to the range
    '''
    base_price = discretize(base_price, price_unit)
    seller_range = (int(base_price * (1. - flexibility)),
            int(base_price * (1. + flexibility)))
    range_size = seller_range[1] - seller_range[0]
    intersection = int(random.choice(intersections) * range_size)
    buyer_upperbound = seller_range[0] + intersection
    buyer_range = (buyer_upperbound - range_size, buyer_upperbound)

    return {SELLER: {'Bottomline': seller_range[0] * price_unit,
                     'Target': seller_range[1] * price_unit},
            BUYER: {'Bottomline': buyer_range[1] * price_unit,
                    'Target': buyer_range[0] * price_unit}
            }

def generate_scenario(schema, base_price, price_unit, intersections, flexibility, listings):
    if listings:
        listing = random.choice(listings)
        base_price = int(listing['price'])
    else:
        listing = None
    kbs = generate_kbs(schema, listing)
    ranges = generate_price_range(base_price, price_unit, intersections, flexibility)
    kbs[BUYER].facts['personal'].update(ranges[BUYER])
    kbs[SELLER].facts['personal'].update(ranges[SELLER])
    return NegotiationScenario(generate_uuid('S'), schema.attributes, kbs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    parser.add_argument('--num-scenarios', help='Number of scenarios to generate', type=int, default=1)
    parser.add_argument('--intersections', nargs='*', type=float, default=[0.2, 0.4, 0.8], help="Intersection of buyer and seller's price range")
    parser.add_argument('--flexibility', type=float, default=0.2, help="Price range")
    parser.add_argument('--text', help="JSON file containing text listings")
    parser.add_argument('--price-unit', default=100, help="Unit for discretizing prices")
    add_scenario_arguments(parser)
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    schema = Schema(args.schema_path)

    if args.text:
        # Real data
        listings = read_json(args.text)
        base_price = None
    else:
        # Simulated data
        listings = None
        base_price = 500

    scenario_list = []
    for i in xrange(args.num_scenarios):
        s = generate_scenario(schema, base_price, args.price_unit, args.intersections, args.flexibility, listings)
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
