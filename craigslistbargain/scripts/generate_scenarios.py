#!/usr/bin/env python

import argparse
import random
import numpy as np
import copy
import re
import langdetect
import os.path
from itertools import izip
from collections import defaultdict

from cocoa.core.schema import Schema
from cocoa.core.scenario_db import ScenarioDB
from cocoa.core.util import generate_uuid, write_json, read_json
import cocoa.options

from core.scenario import Scenario
from core.kb import KB

BUYER = Scenario.BUYER
SELLER = Scenario.SELLER

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
    seller_kb = KB(schema.attributes, {'personal': {'Role': 'seller'}, 'item': seller_item})
    buyer_kb = KB(schema.attributes, {'personal': {'Role': 'buyer'}, 'item': buyer_item})
    kbs = [None, None]
    kbs[BUYER] = buyer_kb
    kbs[SELLER] = seller_kb
    return kbs

def discretize(price, price_unit):
    price = int(price / price_unit)
    return price

def generate_price_range(base_price, price_unit, discounts):
    '''
    base: a middle point to generate the range
    intersections: percentage of intersection relative to the range
    '''
    base_price = discretize(base_price, price_unit)
    seller_target = base_price
    for i in discounts:
        buyer_target = i * seller_target
        # Reverse discretization
        seller_target = int(seller_target * price_unit)
        buyer_target = int(buyer_target * price_unit)
        if seller_target == 0 or buyer_target == 0:
            continue
        yield {
                SELLER: {'Bottomline': None, 'Target': seller_target},
                BUYER: {'Bottomline': None, 'Target': buyer_target},
              }

def generate_scenario(schema, base_price, price_unit, discounts, listings):
    for listing in listings:
        listing = process_listing(listing)
        if listing:
            base_price = int(listing['price'])
            if base_price < price_unit:
                continue
            for ranges in generate_price_range(base_price, price_unit, discounts):
                kbs = generate_kbs(schema, listing)
                kbs[BUYER].facts['personal'].update(ranges[BUYER])
                kbs[SELLER].facts['personal'].update(ranges[SELLER])
                yield Scenario(generate_uuid('S'), listing['post_id'], listing['category'], listing['images'], schema.attributes, kbs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    parser.add_argument('--num-scenarios', help='Number of scenarios to generate', type=int, default=-1)
    parser.add_argument('--discounts', nargs='*', type=float, default=[0.9, 0.7, 0.5], help="Discount that the buyer wants to have")
    parser.add_argument('--scraped-data', required=True, help="Path to scraped data")
    parser.add_argument('--categories', nargs='+', required=True, help="Listing categories, JSON file path is craigslist_<category>.json")
    parser.add_argument('--fractions', nargs='+', required=True, help="Fractions of data from different categories")
    parser.add_argument('--skip', default=0, type=int, help="Skip the first X scenarios")
    cocoa.options.add_scenario_arguments(parser)
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    schema = Schema(args.schema_path)

    listings = [read_json(os.path.join(args.scraped_data, 'craigslist_{}.json'.format(c))) for c in args.categories]
    fractions = np.array([float(x) for x in args.fractions])
    fractions = fractions / np.sum(fractions)

    # Sample listings
    sampled_listings = []
    N = sum([len(l) for l in listings])
    for listing, fraction in izip(listings, fractions):
        n = int(N * fraction)
        print listing[0]['category'], len(listing), fraction, n
        sampled_listings.append(listing[:n])
    listings = [x for l in sampled_listings for x in l]
    N = len(listings)
    inds = np.random.permutation(N)
    listings = [listings[i] for i in inds]

    base_price = None

    scenario_list = []
    price_unit = 1  # Just use the real price
    scenario_generator = generate_scenario(schema, base_price, price_unit, args.discounts, listings)
    for i, s in enumerate(scenario_generator):
        if i % 100 == 0:
            print i
        if i < args.skip:
            continue
        if len(scenario_list) == args.num_scenarios:
            break
        scenario_list.append(s)
    if len(scenario_list) < args.num_scenarios:
        print 'Not enough listings: {} scenarios generated.'.format(len(scenario_list))
    scenario_db = ScenarioDB(scenario_list)
    write_json(scenario_db.to_dict(), args.scenarios_path)

    for i in range(min(10, len(scenario_db.scenarios_list))):
        print '---------------------------------------------------------------------------------------------'
        print '---------------------------------------------------------------------------------------------'
        scenario = scenario_db.scenarios_list[i]
        print "Scenario id: %s" % scenario.uuid
        for agent in (0, 1):
            kb = scenario.kbs[agent]
            kb.dump()

    num_listings_per_category = defaultdict(int)
    for s in scenario_list:
        cat = s.kbs[0].facts['item']['Category']
        num_listings_per_category[cat] += 1
    for k, v in num_listings_per_category.iteritems():
        print k, v
    print '%d scenarios generated' % len(scenario_list)
