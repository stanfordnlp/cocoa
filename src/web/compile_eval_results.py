import argparse
import cPickle as pickle
import json
import numpy as np
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


dialogue_to_responses = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
# Store mean and stddev
dialogue_to_stats = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
# Dialogue ID to agent mapping
dialogue_to_agent_mapping = {}

# Aggregate response scores
for r in responses:
    dialogue_id, _, agent_mapping, _, agent_id, humanlike, correct, strategic, cooperative, fluent = r
    dialogue_to_responses[dialogue_id][agent_id]["humanlike"].append(float(humanlike))
    dialogue_to_responses[dialogue_id][agent_id]["correct"].append(float(correct))
    dialogue_to_responses[dialogue_id][agent_id]["strategic"].append(float(strategic))
    dialogue_to_responses[dialogue_id][agent_id]["cooperative"].append(float(cooperative))
    dialogue_to_responses[dialogue_id][agent_id]["fluent"].append(float(fluent))

    dialogue_to_agent_mapping[dialogue_id] = agent_mapping



# Compute mean/stddev
for dialogue_id, values in dialogue_to_responses.iteritems():
    for agent_id, question_responses in values.iteritems():
        for question, responses in question_responses.iteritems():
            avg = np.array(responses[:5]).mean()
            std = np.array(responses[:5]).std()
            dialogue_to_stats[dialogue_id][agent_id][question].append(avg)
            dialogue_to_stats[dialogue_id][agent_id][question].append(std)


dialogue_eval_info = []
dialogue_eval_info.append(dialogue_to_agent_mapping)
dialogue_eval_info.append(dialogue_to_responses)
dialogue_eval_info.append(dialogue_to_stats)

# Dump dialogue to average
with open("dialogue_eval_info.json", "w") as f:
    json.dump(dialogue_eval_info, f)
