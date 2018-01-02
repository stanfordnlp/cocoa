'''
Combine different batches of data (transcripts.json and surveys.json).
'''

import argparse
import os
from cocoa.core.util import read_json, write_json

parser = argparse.ArgumentParser()
parser.add_argument('--paths', nargs='+', help='Paths to transcripts directories')
parser.add_argument('--output', help='Output directory')
args = parser.parse_args()

all_chats = []
# survey data structure: [{}, {}]
all_surveys = [{}, {}]

for d in args.paths:
    transcript_file = os.path.join(d, 'transcripts/transcripts.json')
    survey_file = os.path.join(d, 'transcripts/surveys.json')

    chats = read_json(transcript_file)
    all_chats.extend(chats)

    surveys = read_json(survey_file)
    for i, s in enumerate(surveys):
        all_surveys[i].update(s)
    print "Combined data from {}".format(d)

output_dir = args.output + '/transcripts'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
write_json(all_chats, os.path.join(output_dir, 'transcripts.json'))
write_json(all_surveys, os.path.join(output_dir, 'surveys.json'))
