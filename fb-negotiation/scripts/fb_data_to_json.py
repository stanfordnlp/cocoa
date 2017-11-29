import json
import argparse
import copy
import time as tm

def fill_scenario(knowledge_bases, stamp):
  attr_item = { 'entity': True, 'unique': False, 'value_type': 'item',
                'name': 'Name', 'multivalued': False }
  attr_integer = { 'entity': True, 'unique': False, 'value_type': 'integer',
                'name': 'Name', 'multivalued': False }
  scenario_dict = {
    'attributes': [attr_item, attr_integer, attr_integer],
    'uuid': 'scenario_uuid_{}'.format(stamp),
    'kbs': knowledge_bases
  }
  return scenario_dict

def fit_into_transcripts(data, counter):
  stamp = str(counter).zfill(3)
  scenario_dict = fill_scenario( data['kbs'], stamp )
  transcript = {
    'uuid': 'uuid_{}'.format(stamp),
    'scenario': scenario_dict,
    'agents_info': {'config': [8, 5, 10]},
    'scenario_uuid': 'scenario_uuid_{}'.format(stamp),
    'agents': {0: 'human', 1: 'human'},
    'outcome': data['outcome'],
    'events': data['events']
  }
  return transcript

def make_kb(tokens):
  scores = [int(t) for t in tokens if not t.startswith("<")]
  kb = [
    {'Count': scores[0], 'Name': 'book', 'Value': scores[1]},
    {'Count': scores[2], 'Name': 'hat', 'Value': scores[3]},
    {'Count': scores[4], 'Name': 'ball', 'Value': scores[5]}
  ]
  return kb

def break_into_turns(tokens):
  turns = []
  event = []
  for token in tokens:
    if token in ['<dialogue>', '</dialogue>']:
      continue
    if token == '<eos>':
      turns.append(event)
      event = []
    else:
      event.append(token)
  return turns

def calculate_rewards(counts, scenario):
  reward = 0
  for idx, item in enumerate(['book', 'hat', 'ball']):
    value = scenario[idx]['Value']
    count = counts[item]
    reward += value * count
  return reward

def calculate_splits(tokens):
  splits = [int(t.split("=")[1]) for t in tokens if not t.startswith("<")]
  items = { 0: {'book': splits[0], 'hat': splits[1], 'ball': splits[2]},
            1: {'book': splits[3], 'hat': splits[4], 'ball': splits[5]}   }
  return items

def find_results(tokens, scenario):
  deal_validity = True
  agreement = True
  if tokens[1].startswith('item'):
    items = calculate_splits(tokens)
    reward_0 = calculate_rewards(items[0], scenario[0])
    reward_1 = calculate_rewards(items[1], scenario[1])
  else:
    items = {0:{'book':0, 'hat':0, 'ball':0}, 1:{'book':0, 'hat':0, 'ball':0}}
    reward_0, reward_1 = (0,0)
    if tokens[1] == '<disagree>':
      deal_validity = False
    elif tokens[1] in ['<no_agreement>', '<disconnect>']:
      agreement = False
    else:
      print "Something went wrong: {}".format(tokens)

  outcome_dict = {
    'valid_deal': deal_validity,
    'item_split': items,
    'reward': {0: reward_0, 1: reward_1},
    'agreed': agreement
  }
  return outcome_dict

def populate_event(agent_id, action, data):
  timestamp = tm.time()
  event = {
    'start_time': timestamp,
    'agent': agent_id,
    'template': None,
    'time': timestamp,
    'action': action,
    'data': data
  }
  return event

def parse_events(dialogue, outcome):
  turns = break_into_turns(dialogue)
  event_list = []
  for action in turns:
    agent_id = 0 if action.pop(0) == 'YOU:' else 1 # agent == THEM:
    msg_event = populate_event(agent_id, 'message', " ".join(action))
    event_list.append(msg_event)
  if outcome['valid_deal'] and outcome['agreed']:
    items = outcome['item_split']
  else:
    items = [{}, {}]
  event_list.append( populate_event(0, 'select', items[0]) )
  event_list.append( populate_event(1, 'select', items[1]) )
  return event_list

def process_data(examples):
  ready = []
  counter = 0
  for example in examples:
    cleaned = {}
    cleaned['kbs'] = [ make_kb(example['input']), make_kb(example['partner']) ]
    assert cleaned['kbs'][0][2]['Count'] == cleaned['kbs'][1][2]['Count']
    cleaned['outcome'] = find_results(example['output'], cleaned['kbs'])
    cleaned['events'] = parse_events(example['dialogue'], cleaned['outcome'])
    ready.append( fit_into_transcripts(cleaned, counter) )
    counter += 1
  return ready

def process_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--split-type', default='test',
    help='Path to the file containing the raw dialog input')
  parser.add_argument('--output', default='transformed',
    help='Path to the output JSON scenario file')
  return parser.parse_args()

def extract_examples(in_path, out_path):
  examples = []
  with open(in_path) as file:
    for line in file:
      example = {'input': [], 'dialogue': [], 'output': [], 'partner': []}
      tokens = line.strip().split()
      stage = 'input'
      for token in tokens:
        example[stage].append(token)
        if token == "</input>":
          stage = 'dialogue'
        elif token == "</dialogue>":
          stage = 'output'
        elif token == "</output>":
          stage = 'partner'
      examples.append(example)
  print "Extracted {} examples from {}".format(len(examples), in_path)
  return process_data(examples)

if __name__ == "__main__":
  args = process_args()
  in_path = 'data/{}.txt'.format(args.split_type)
  out_path = 'data/{0}{1}.json'.format(args.output, args.split_type)
  ready = extract_examples(in_path, out_path)
  json.dump(ready, open(out_path, "w"))
