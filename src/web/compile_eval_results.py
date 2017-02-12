import argparse
import json
import numpy as np
import sqlite3

from collections import defaultdict
from statsmodels.stats.inter_rater import fleiss_kappa


parser = argparse.ArgumentParser()
parser.add_argument("--db-path", type=str, help="path to db to output results from")
args = vars(parser.parse_args())


def bin(ratings):
    """
    Bin provided ratings into len(ratings) bins based on counts of ratings,
    assumed to be discretized
    :param ratings:
    :return:
    """
    binned = np.zeros(5)
    for r in ratings:
        r = int(r)
        binned[r-1] += 1
    return binned


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

dialogue_to_scenario = {}


# Aggregate response scores
for r in responses:
    dialogue_id, scenario_id, agent_mapping, _, agent_id, humanlike, correct, cooperative, fluent, humanlike_text, correct_text, cooperative_text, fluent_text = r
    dialogue_to_responses[dialogue_id][agent_id]["humanlike"].append(float(humanlike))
    dialogue_to_responses[dialogue_id][agent_id]["correct"].append(float(correct))
    dialogue_to_responses[dialogue_id][agent_id]["cooperative"].append(float(cooperative))
    dialogue_to_responses[dialogue_id][agent_id]["fluent"].append(float(fluent))

    dialogue_to_responses[dialogue_id][agent_id]["humanlike_text"].append(humanlike_text)
    dialogue_to_responses[dialogue_id][agent_id]["correct_text"].append(correct_text)
    dialogue_to_responses[dialogue_id][agent_id]["cooperative_text"].append(cooperative_text)
    dialogue_to_responses[dialogue_id][agent_id]["fluent_text"].append(fluent_text)

    dialogue_to_agent_mapping[dialogue_id] = agent_mapping
    dialogue_to_scenario[dialogue_id] = scenario_id


# Compute mean/stddev
for dialogue_id, values in dialogue_to_responses.iteritems():
    for agent_id, question_responses in values.iteritems():
        question_arr = []
        for question, responses in question_responses.iteritems():
            if question not in ["humanlike_text", "correct_text", "strategic_text", "cooperative_text", "fluent_text"]:
                responses = np.array(responses[:5])
                question_arr.append(bin(responses))

                avg = responses.mean()
                median = np.median(responses)
                std = responses.std()

                dialogue_to_stats[dialogue_id][agent_id][question].append(avg)
                dialogue_to_stats[dialogue_id][agent_id][question].append(median)
                dialogue_to_stats[dialogue_id][agent_id][question].append(std)

        question_arr = np.array(question_arr)
        kappa = fleiss_kappa(question_arr)
        dialogue_to_stats[dialogue_id][agent_id]["kappa"].append(kappa)


dialogue_eval_info = []
dialogue_eval_info.append(dialogue_to_agent_mapping)
dialogue_eval_info.append(dialogue_to_responses)
dialogue_eval_info.append(dialogue_to_stats)

scenario_id_to_mappings = defaultdict(list)


# Name of eval results file
eval_results_file = None

# Dump dialogue to average
with open(eval_results_file, "w") as f:
    json.dump(dialogue_eval_info, f)
