import json
import copy
from cocoa.core.scenario_db import ScenarioDB
from cocoa.core.schema import Schema
from cocoa.core.util import generate_uuid, write_json, read_json
from core.kb import KB
from core.scenario import Scenario

root_path = "data/"
scenario_list = []

schema = Schema(root_path + "bookhatball-schema.json")

scenarios = []
with open(root_path + "selfplay.txt") as fin:
    kbs = []
    names = ['book', 'hat', 'ball']
    for line in fin:
        ints = [int(x) for x in line.strip().split()]
        kb = KB.from_ints(schema.attributes, names, ints)
        kbs.append(kb)
        if len(kbs) == 2:
            scenario = Scenario(generate_uuid("FB"), schema.attributes, kbs)
            scenarios.append(scenario)
            kbs = []

scenario_db = ScenarioDB(scenarios)
write_json(scenario_db.to_dict(), root_path + "test-scenarios.json")
