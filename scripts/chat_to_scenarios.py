import argparse
from cocoa.core.util import read_json, write_json
from cocoa.core.scenario_db import ScenarioDB
from cocoa.core.schema import Schema
from core.scenario import Scenario

parser = argparse.ArgumentParser()
parser.add_argument('--chats')
parser.add_argument('--scenarios')
parser.add_argument('--max', type=int)
args = parser.parse_args()

chats = read_json(args.chats)
scenarios = []
n = args.max or len(chats)
for chat in chats[:n]:
    scenarios.append(Scenario.from_dict(None, chat['scenario']))
scenario_db = ScenarioDB(scenarios)
write_json(scenario_db.to_dict(), args.scenarios)
