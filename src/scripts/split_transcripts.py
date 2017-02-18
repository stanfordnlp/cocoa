'''
Split human-bot/human mixed transcripts to multiple transcripts of chats between human
and a single type of partner.
'''

import argparse
import os
from collections import defaultdict
from src.basic.util import read_json, write_json

parser = argparse.ArgumentParser()
parser.add_argument('--transcripts', help='Path to transcritps of mixed partners')
parser.add_argument('--output', help='Output directories')
args = parser.parse_args()

chats = read_json(args.transcripts)
chats_by_agents = defaultdict(list)
scenario_agents = defaultdict(set)
for chat in chats:
    agents = chat['agents']
    if agents['0'] == 'human':
        agents = (agents['0'], agents['1'])
    else:
        agents = (agents['1'], agents['0'])
    chats_by_agents[agents].append(chat)

    scenario_id = chat['scenario_uuid']
    scenario_agents[scenario_id].add(agents)

# Only keep scenarios with all 4 agents
scenario_subset = set([s for s, a in scenario_agents.iteritems() if len(a) == 4])
print 'Number of scenarios:', len(scenario_subset)

for agents, chats in chats_by_agents.iteritems():
    chats = [c for c in chats if c['scenario_uuid'] in scenario_subset]
    print agents, len(chats)
    path = os.path.join(args.output, '%s_transcripts.json' % '-'.join(agents))
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    write_json(chats, path)
