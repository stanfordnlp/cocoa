'''
Output all scenarios in a chat transcript.
'''

import sys
import argparse
from src.basic.util import read_json, write_json
from src.basic.scenario_db import Scenario, ScenarioDB
from src.basic.schema import Schema

parser = argparse.ArgumentParser()
parser.add_argument('--chats')
parser.add_argument('--scenarios')
parser.add_argument('--schema-path')
args = parser.parse_args()

chats = read_json(args.chats)
schema = Schema(args.schema_path)
scenarios = []
for chat in chats:
    scenarios.append(Scenario.from_dict(schema, chat['scenario']))
scenario_db = ScenarioDB(scenarios)
write_json(scenario_db.to_dict(), args.scenarios)
