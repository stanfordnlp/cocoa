#!/usr/bin/env python

import argparse
import random
import numpy as np
import copy
import re
import langdetect
from itertools import izip
from src.basic.schema import Schema
from src.basic.scenario_db import NegotiationScenario, ScenarioDB, add_scenario_arguments
from src.basic.util import generate_uuid, write_json, read_json
from src.basic.kb import NegotiationKB
from itertools import izip_longest

private_attr = ['Laundry', 'Pet', 'Built data', 'Neighborhood']
BUYER = NegotiationScenario.BUYER
SELLER = NegotiationScenario.SELLER

def is_valid_line(line):
    if 'contact' in line.lower():
        return False
    if not re.search(r'\.|\!|\,', line) and len(line.split()) > 15:
        return False
    if re.search(r'\$\s*\d+', line):
        return False
    try:
        if langdetect.detect(line) != 'en':
            return False
    except langdetect.lang_detect_exception.LangDetectException:
        return True
    return True

def process_listing(listing):
    if listing['category'] == 'car' and listing['price'] < 3000:
        return None

    lines = []
    for line in listing['description']:
        if not is_valid_line(line):
            continue
        lines.append(line)

    num_words = sum([len(line.split()) for line in lines])
    if num_words < 20 or num_words > 200:
        return None

    listing['description'] = lines

    return listing

def generate_kbs(schema, listing):
    buyer_item, seller_item = {}, {}
    for attr in schema.attributes:
        if attr.name in ('Role', 'Target', 'Bottomline'):
            continue
        l = listing[attr.name.lower()]
        if attr.name == 'Description':
            # NOTE: Buyer only sees the first half
            N = max(1, len(l) / 2)
            buyer_item[attr.name] = l[:N]
        else:
            buyer_item[attr.name] = l
        seller_item[attr.name] = listing[attr.name.lower()]
    seller_kb = NegotiationKB(schema.attributes, {'personal': {'Role': 'seller'}, 'item': seller_item})
    buyer_kb = NegotiationKB(schema.attributes, {'personal': {'Role': 'buyer'}, 'item': buyer_item})
    kbs = [None, None]
    kbs[BUYER] = buyer_kb
    kbs[SELLER] = seller_kb
    return kbs

def discretize(price, price_unit):
    price = int(price / price_unit)
    return price

def generate_price_range(base_price, price_unit, intersections, flexibility=0.2):
    '''
    base: a middle point to generate the range
    intersections: percentage of intersection relative to the range
    '''
    base_price = discretize(base_price, price_unit)
    seller_bottomline = base_price * (1. - flexibility)
    seller_range = base_price * flexibility
    for i in intersections:
        intersection = i * seller_range
        buyer_bottomline = seller_bottomline + intersection
        yield {
                SELLER: {'Bottomline': int(seller_bottomline * price_unit), 'Target': None},
                BUYER: {'Bottomline': int(buyer_bottomline * price_unit), 'Target': None},
                'intersection': i,
              }

def generate_scenario(schema, base_price, price_unit, intersections, flexibility, listings):
    for listing in listings:
        listing = process_listing(listing)
        if listing:
            base_price = int(listing['price'])
            if base_price < price_unit:
                continue
            for ranges in generate_price_range(base_price, price_unit, intersections, flexibility):
                kbs = generate_kbs(schema, listing)
                kbs[BUYER].facts['personal'].update(ranges[BUYER])
                kbs[SELLER].facts['personal'].update(ranges[SELLER])
                yield NegotiationScenario(generate_uuid('S'), listing['post_id'], listing['category'], listing['images'], schema.attributes, kbs, ranges['intersection'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    parser.add_argument('--num-scenarios', help='Number of scenarios to generate', type=int, default=-1)
    parser.add_argument('--intersections', nargs='*', type=float, default=[0.2, 0.4, 0.6, 0.8], help="Intersection of buyer and seller's price range")
    parser.add_argument('--flexibility', type=float, default=0.2, help="Price range")
    parser.add_argument('--scraped-data', nargs='+', required=True, help="JSON file containing text listings")
    parser.add_argument('--price-unit', default=10, help="Unit for discretizing prices")
    add_scenario_arguments(parser)
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    schema = Schema(args.schema_path)

    listings = [read_json(data) for data in args.scraped_data]
    # Interleave listings from different categories so we have a balanced scenario set
    listings = [x for l in izip_longest(*listings, fillvalue=None) for x in l if x is not None]
    base_price = None

    scenario_list = []
    scenario_generator = generate_scenario(schema, base_price, args.price_unit, args.intersections, args.flexibility, listings)
    for i, s in enumerate(scenario_generator):
        if i == args.num_scenarios:
            break
        scenario_list.append(s)
    scenario_db = ScenarioDB(scenario_list)
    write_json(scenario_db.to_dict(), args.scenarios_path)

    for i in range(min(100, len(scenario_db.scenarios_list))):
        # print '---------------------------------------------------------------------------------------------'
        # print '---------------------------------------------------------------------------------------------'
        scenario = scenario_db.scenarios_list[i]
        # print "Scenario id: %s" % scenario.uuid, scenario.intersection
        for agent in (0, 1):
            kb = scenario.kbs[agent]
            # kb.dump()

    print '%d scenarios generated' % len(scenario_list)
