import argparse
import cPickle as pickle
import json
import numpy
import sqlite3

from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument("--db-path", type=str, help="path to db to output results from")
args = vars(parser.parse_args())

conn = sqlite3.connect(args["db_path"])
curs = conn.cursor()

# Get all responses
curs.execute("SELECT * FROM Responses")
responses = curs.fetchall()

# Get all completed dialogues
curs.execute("SELECT * FROM CompletedDialogues")
completed_dialogues = curs.fetchall()

dialogue_to_average = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))

# Dialogue ID -> agent id -> num responses
dialogue_to_num_responses = defaultdict(lambda : defaultdict(float))

# Aggregate response scores
for r in responses:
    dialogue_id, _, _, agent_id, humanlike, correct, strategic, cooperative, fluent = r
    dialogue_to_average[dialogue_id][agent_id]["humanlike"] += float(humanlike)
    dialogue_to_average[dialogue_id][agent_id]["correct"] += float(correct)
    dialogue_to_average[dialogue_id][agent_id]["strategic"] += float(strategic)
    dialogue_to_average[dialogue_id][agent_id]["cooperative"] += float(cooperative)
    dialogue_to_average[dialogue_id][agent_id]["fluent"] += float(fluent)

    dialogue_to_num_responses[dialogue_id][agent_id] += 1

# Normalize response scores
for dialogue_id, values in dialogue_to_average.iteritems():
    for agent_id, scores in values.iteritems():
        num_responses = dialogue_to_num_responses[dialogue_id][agent_id]
        # humanlike, correct, fluent, etc.
        for metric in scores.keys():
            dialogue_to_average[dialogue_id][agent_id][metric] /= num_responses


# Dump dialogue to average
with open("dialogue_to_average.json", "w") as f:
    json.dump(dict(dialogue_to_average), f)
