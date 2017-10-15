import json
import argparse
import copy
from cocoa.core.scenario_db import ScenarioDB
from cocoa.core.schema import Schema
from cocoa.core.util import generate_uuid, write_json, read_json
from core.kb import KB
from core.scenario import Scenario

parser = argparse.ArgumentParser()
parser.add_argument('--schema-path')
parser.add_argument('--scenario-ints-file', help='Path to the file containing 6 integers per line that describes the scenario')
parser.add_argument('--output', help='Path to the output JSON scenario file')
args = parser.parse_args()

schema = Schema(args.schema_path)

scenarios = []
with open(args.scenario_ints_file) as fin:
    kbs = []
    names = ['book', 'hat', 'ball']
    for line in fin:
        ints = [int(x) for x in line.strip().split()]
        kb = KB.from_ints(schema.attributes, names, ints)
        kbs.append(kb)
        if len(kbs) == 2:
            if kbs[0].item_counts != kbs[1].item_counts:
                del kbs[0]
                continue
            assert kbs[0].item_counts == kbs[1].item_counts
            scenario = Scenario(generate_uuid("FB"), schema.attributes, kbs)
            scenarios.append(scenario)
            kbs = []

scenario_db = ScenarioDB(scenarios)
write_json(scenario_db.to_dict(), args.output)
