import random
import json
import string
import sys
import time as tm
import argparse

def generate_uuid(prefix):
  return prefix + '_' + ''.join([random.choice(string.digits + string.letters) for _ in range(16)])

def populate_events(dialogue, item_split):
  events = []
  timestep = 0
  turns = []
  token_turns = []

  #if dialogue[0] == "THEM:":
  #  agents = {"THEM:": 0, "YOU:": 1}
  #elif dialogue[0] == "YOU:":
  #  agents = {"THEM:": 1, "YOU:": 0}
  #else:
  #  print "ERROR:", dialogue
  #  sys.exit()
  agents = {"YOU:": 0, "THEM:": 1}

  for token in dialogue:
    if token in ["THEM:", "YOU:"]:
      if len(token_turns) > 0:
        # we are in a new turn, close the previous turn
        turns.append(token_turns)
      # start a new turn
      token_turns = [agents[token]]
    elif token not in ["<eos>"]:
      token_turns.append(token)
  # grab the last selection token
  if len(token_turns) > 1:
    turns.append(token_turns)

  for turn in turns:
    if turn[1] == "<selection>":
      # item_split[0] is agent=0 is "YOU"
      event = create_one_event(turn[0], timestep, item_split[turn[0]], "select")
      events.append(event)
      timestep += 1
      event = create_one_event(1-turn[0], timestep, item_split[1-turn[0]], "select")
      events.append(event)
      timestep += 1
    else:  # message actions
      event = create_one_event(turn[0], timestep, turn[1:])
      events.append(event)
      timestep += 1

  return events

def create_one_event(agent, timestep, data, action="message"):
  text = " ".join(data) if action == "message" else data
  event = {
    "start_time": tm.time(),
    "agent": agent,
    "template": None,
    "time": timestep,
    "action": action,
    "data": text
  }
  return event

def create_scenario(data, sid):

  attr = [{"entity": True, "unique": False, "value_type": "item",
              "name": "Name", "multivalued": False },
          {"entity": True, "unique": False, "value_type": "integer",
              "name": "Name", "multivalued": False },
          {"entity": True, "unique": False, "value_type": "integer",
              "name": "Name", "multivalued": False }
          ]
  kbs = create_knowledge_bases(data)
  return {"attributes": attr, "uuid": sid, "kbs": kbs}

def create_knowledge_bases(data):
  my_input = data['input']
  part_input = data['part_input']

  my_kb = [
    {"Count": int(my_input[0]), "Name": "book", "Value": int(my_input[1])},
    {"Count": int(my_input[2]), "Name": "hat", "Value": int(my_input[3])},
    {"Count": int(my_input[4]), "Name": "ball", "Value": int(my_input[5])}
  ]
  part_kb = [
    {"Count": int(part_input[0]), "Name": "book", "Value": int(part_input[1])},
    {"Count": int(part_input[2]), "Name": "hat", "Value": int(part_input[3])},
    {"Count": int(part_input[4]), "Name": "ball", "Value": int(part_input[5])}
  ]

  return [my_kb, part_kb]

def read_file(in_filename):
  in_lines = []
  with open(in_filename, 'r') as infile:
    for line in infile:
      in_lines.append(line.strip())
  return in_lines

def store_lines(outfile, cleaned):
  with open(outfile, 'w') as file:
    json.dump(cleaned, file)
  print("Completed saving {0} lines into {1}".format(len(cleaned), outfile) )

def remove_speaker_id(line):
    line = [x for x in line if x not in ('YOU:', 'THEM:')]
    return line

def parse_lines(raw):
  parsed = []
  for line in raw:
    result = parse_one_line(line.split())
    # Skip invalid chat
    if result['output'][0] in ["<disagree>", "<disconnect>", "<no_agreement>"]:
        continue
    # Skip repeated chat
    if parsed and remove_speaker_id(result['dialogue']) == \
            remove_speaker_id(parsed[-1]['dialogue']):
        continue
    parsed.append(result)
  return parsed

def parse_one_line(tokens):
  result = {
    "input": get_tag(tokens, 'input'),
    "dialogue": get_tag(tokens, 'dialogue'),
    "output": get_tag(tokens, 'output'),
    "part_input": get_tag(tokens, 'partner_input'),
  }
  return result

def get_tag(tokens, tag):
  # the index right after the start tag
  start = tokens.index('<' + tag + '>') + 1
  # the index of the ending tag
  stop = tokens.index('</' + tag + '>')
  return tokens[start:stop]

def create_outcome(data):
  out = data['output']

  if out[0] in ["<disagree>", "<disconnect>", "<no_agreement>"]:
    outcome = {
      "valid_deal": False,
      "item_split": [
        {"book": 0, "hat": 0, "ball": 0},
        {"book": 0, "hat": 0, "ball": 0}
      ],
      "reward": 0,
      "agreed": False
    }
    return outcome

  my_book = int(out[0].split("=")[1])
  my_hat = int(out[1].split("=")[1])
  my_ball = int(out[2].split("=")[1])
  part_book = int(out[3].split("=")[1])
  part_hat = int(out[4].split("=")[1])
  part_ball = int(out[5].split("=")[1])

  outcome = {
    "valid_deal": determine_validity(data),
    "item_split": [
      {"book": my_book, "hat": my_hat, "ball": my_ball},
      {"book": part_book, "hat": part_hat, "ball": part_ball}
    ],
    "reward": calculate_reward(data),
    "agreed": data['dialogue'][-1] == "<selection>"
  }

  return outcome

def calculate_reward(data):
  out = data['output']
  my_input = data['input']
  part_input = data['part_input']

  my_book = int(out[0].split("=")[1])
  my_hat = int(out[1].split("=")[1])
  my_ball = int(out[2].split("=")[1])
  part_book = int(out[3].split("=")[1])
  part_hat = int(out[4].split("=")[1])
  part_ball = int(out[5].split("=")[1])

  my_reward = (my_book * int(my_input[1])) + (my_hat * int(my_input[3])) + (my_ball * int(my_input[5]))
  part_reward = (part_book * int(part_input[1])) + (part_hat * int(part_input[3])) + (part_ball * int(part_input[5]))

  reward = {
    0: int(my_reward),
    1: int(part_reward)
  }

  return reward

def determine_validity(data):
  out = data['output']
  my_input = data['input']

  total_book = int(my_input[0])
  total_hat = int(my_input[2])
  total_ball = int(my_input[4])

  my_book = int(out[0].split("=")[1])
  my_hat = int(out[1].split("=")[1])
  my_ball = int(out[2].split("=")[1])
  part_book = int(out[3].split("=")[1])
  part_hat = int(out[4].split("=")[1])
  part_ball = int(out[5].split("=")[1])

  if (my_book + part_book != total_book):
    return False
  elif (my_hat + part_hat != total_hat):
    return False
  elif (my_ball + part_ball != total_ball):
    return False
  else:
    return True

def process_lines(parsed):
  cleaned = []
  for i, data in enumerate(parsed):
    uuid = generate_uuid("E")
    scenario_id = generate_uuid("FB")
    world = {
      "uuid": uuid,
      "scenario": create_scenario(data, scenario_id),
      "agents_info": {},
      "scenario_uuid": scenario_id,
      "agents": {0: "human", 1: "human"},
      "outcome": create_outcome(data),
    }
    world["events"] = populate_events(data["dialogue"], world["outcome"]['item_split'])
    cleaned.append(world)

  for world in cleaned:
      print '-'*20
      for event in world["events"]:
          print event
      print world['outcome']
      print '-'*20

  return cleaned

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    random.seed(0)
    for split in ("val", "test", "train"):
        in_filename = "{}/{}.txt".format(args.input_dir, split)
        out_filename = "{}/{}.json".format(args.output_dir, split)

        raw_in = read_file(in_filename)
        parsed = parse_lines(raw_in)
        cleaned = process_lines(parsed)
        store_lines(out_filename, cleaned)
