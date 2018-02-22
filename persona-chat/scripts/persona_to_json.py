import random
import json
import string
import pdb

def generate_uuid(prefix):
  return prefix + '_' + ''.join([random.choice(string.digits + string.letters) for _ in range(16)])

def populate_events(messages):
  events = []
  timestep = 0
  for message_pair in messages:
    user_msg, agent_msg = message_pair
    user_event = {"start_time": None, "agent": 0, "template": None,
          "time": timestep, "action": "message", "data": user_msg }
    timestep += 1
    events.append(user_event)

    agent_event = {"start_time": None, "agent": 1, "template": None,
          "time": timestep, "action": "message", "data": agent_msg }
    timestep += 1
    events.append(agent_event)
  return events

def build_kbs(personas):
  # inputs: a list of dicts, where each dict has keys Owner, Description
  # outputs: a list of two lists, where list 0 = your persona, 1 = their persona
  yours = []
  partners = []
  for p in personas:
    if p["Owner"] == "your persona":
      yours.append(p["Description"])
    elif p["Owner"] == "partner's persona":
      partners.append(p["Description"])

  return [yours, partners]

def create_scene(personas, sid):
  attr = [{"entity": True, "unique": False, "value_type": "owner",
              "name": "Persona", "multivalued": False }]
  return {"attributes": attr, "uuid": sid, "kbs": build_kbs(personas)}

def create_world(data, scenario):
  uuid = generate_uuid("E")
  scenario_id = scenario["uuid"]
  world = {
    "uuid": uuid,
    "scenario": scenario,
    "agents_info": None,
    "scenario_uuid": scenario_id,
    "agents": None,
    "outcome": {"reward": 0, "done": False},
    "events": populate_events(data["messages"])
  }
  return world

def parse_line(number, text, file_type):
  if file_type == "none_original":
    data_type = "message"     # there are no personas, so every line is message
  else:
    data_type = "persona" if text.split(" ")[1] == "persona:" else "message"

  if data_type == "persona":
    owner, description = text.split(":")
    persona = {"Owner": owner, "Description": description.rstrip()}
    return "personas", persona
  elif data_type == "message":
    items = text.split("\t")
    user_utterance = items[0].lstrip()
    agent_utterance = items[1]
    return "messages", (user_utterance, agent_utterance)

if __name__ == "__main__":
  split_types = ["test_", "train_", "valid_"]
  file_types = ["original", "revised"] #, "none_original"
            #, "other_original", "other_revised", "self_original", "self_revised"]
  # pdb.set_trace()
  for split in split_types:
    for file in file_types:
      in_filename = split + "both_" + file + ".txt"
      scenario_filename = split + file + "_scenarios.json"
      examples_filename = split + file + "_examples.json"

      scenarios = []
      all_data = []
      world_data = {"personas": [], "messages": [], "active": False}

      with open(in_filename) as infile:
        for line in infile:
          line_number = int(line[:2])
          line_text = line[2:-1].strip()
          # save the world and start over
          if world_data["active"] and (line_number == 1):
            scenario = create_scene(world_data["personas"], generate_uuid("PC"))
            scenarios.append(scenario)
            all_data.append( create_world(world_data, scenario) )
            world_data = {"personas": [], "messages": [], "active": False}
          data_type, results = parse_line(line_number, line_text, file)
          world_data[data_type].append(results)
          world_data["active"] = True

        scenario = create_scene(world_data["personas"], generate_uuid("PC"))
        scenarios.append(scenario)
        all_data.append( create_world(world_data, scenario) )

      with open(scenario_filename, 'w') as outfile:
        json.dump(scenarios, outfile)
      with open(examples_filename, 'w') as outfile:
        json.dump(all_data, outfile)
    print("Completed {} files".format(split) )
