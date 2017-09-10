import json
import copy
from cocoa.core.util import generate_uuid

root_path = "fb-negotiation/data/"
scenario_list = []

with open(root_path + "bookhatball-schema.json") as schema_file:
    schema = json.load(schema_file)

with open(root_path + "selfplay.txt") as file:
  for line in file:
    lineA = line.strip('\n')
    line = next(file)
    lineB = line.strip('\n')

    scenario = copy.deepcopy(schema)
    scenario["uuid"] = generate_uuid("FB")

    kb_A = [int(x) for x in lineA.split(" ")]
    kb_B = [int(x) for x in lineB.split(" ")]
    kb1 = { "Role": "first",
      "Item_counts": {"book": kb_A[0], "hat": kb_A[2], "ball":kb_A[4]},
      "Item_values": {"book": kb_A[1], "hat": kb_A[3], "ball":kb_A[5]},
    }
    kb2 = { "Role": "second",
      "Item_counts": {"book": kb_B[0], "hat": kb_B[2], "ball":kb_B[4]},
      "Item_values": {"book": kb_B[1], "hat": kb_B[3], "ball":kb_B[5]},
    }
    scenario["kbs"] = [kb1, kb2]

    scenario_list.append(scenario)

with open(root_path + "test-scenarios.json", "w") as outfile:
    json.dump(scenario_list, outfile)