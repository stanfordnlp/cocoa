from src.basic.event import Event
from src.basic.scenario_db import Scenario
from src.basic.dataset import Example
import random
from argparse import ArgumentParser
import os
import json
import codecs

__author__ = 'anushabala'


def convert_transcript(path):
    infile = codecs.open(path, 'r', encoding='utf-8')
    events = []
    scenario_id = None
    for line in infile.readlines():
        line = line.strip().split('\t')
        if len(line) <=2:
            # if length == 0 or 1, invalid line
            # if length == 2, end of chat, don't add any events
            pass
        elif len(line) == 3:
            # Line logs user ID - do nothing
            # Line has format timestamp \t scenario \t User X has user ID user_id_key
            pass
        else:
            time = line[0]
            scenario_id = line[1]
            user_idx = line[2]
            agent = 1 if '1' in user_idx else 0

            if line[3] == 'joined':
                event = Event(agent, time, 'join_chat', data=None)
            elif line[3].startswith('Selected'):
                selection = line[4]
                data = {'Name': selection}
                event = Event(agent, time, 'select', data)
            else:
                event = Event(agent, time, 'message', data=line[3])

            events.append(event)

    infile.close()

    # Create a dummy Scenario object with the correct scenario_id (because the Example constructor expects one)
    scenario = Scenario(scenario_id, None)
    # todo this doesn't actually check whether the conversation was successfully completed or not
    outcome = {'reward': 1}

    return Example(scenario, scenario_id, events, outcome)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path', nargs='+',
                        help='Path to directory containing chat transcripts in old .txt format. Multiple paths can be'
                             'provided to consolidate transcripts from multiple directories.')
    parser.add_argument('--output', default='friends-turk-examples.json')

    args = parser.parse_args()

    examples = []

    for dirname in args.path:
        for f in os.listdir(dirname):
            fname = os.path.join(dirname, f)
            examples.append(convert_transcript(fname))

    out = open(args.output, 'w')
    json.dump([e.to_dict() for e in examples], out)
    out.close()

