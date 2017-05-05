__author__ = 'anushabala'

from src.scripts.dataset_statistics import get_total_statistics, print_group_stats
from argparse import ArgumentParser
from src.basic.schema import Schema
from src.basic.scenario_db import ScenarioDB
from src.model.preprocess import tokenize
import json
from src.basic.util import read_json
import os
from scipy import stats


def get_dialogue_tokens(transcript):
    all_tokens = {0: [], 1: []}
    for event in transcript["events"]:
        if event["action"] == 'message':
            tokens = tokenize(event["data"])
            all_tokens[event["agent"]].extend(tokens)
    return all_tokens


def get_normalized_overlap(description, observed_tokens):
    mentions = 0.
    for token in observed_tokens:
        if token[0] in description:
            mentions += 1.
    if len(observed_tokens) == 0:
        return 0
    return mentions / len(observed_tokens)


def description_overlap(transcript):
    all_desc_tokens = [set(tokenize(s)) for s in transcript["scenario"]["kbs"][0]["item"]["Description"]]
    description= set()
    for s in all_desc_tokens:
        description.update(s)

    description.update(set(tokenize(transcript["scenario"]["kbs"][0]["item"]["Title"])))
    agent_tokens = get_dialogue_tokens(transcript)

    overlap = {0: get_normalized_overlap(description, agent_tokens[0]),
               1: get_normalized_overlap(description, agent_tokens[1])}

    return overlap


def compute_avg_description_overlap(transcripts):
    num_agents = 0
    total_overlap = 0.
    for t in transcripts:
        overlap = description_overlap(t)
        total_overlap += overlap[0] + overlap[1]
        num_agents += 2

    return total_overlap/num_agents


def get_overlap_correlation(transcripts, surveys, questions=("persuasive", "negotiator")):
    avg_overlaps = []
    ratings = dict((q,[]) for q in questions)
    for t in transcripts:
        cid = t["uuid"]
        overlap = description_overlap(t)
        if cid not in surveys.keys():
            continue
        chat_survey = surveys[cid]
        if "0" in chat_survey.keys() and len(chat_survey["0"].keys()) > 0:
            avg_overlaps.append(overlap[0])
            for q in questions:
                ratings[q].append(chat_survey["0"][q])
        if "1" in chat_survey.keys() and len(chat_survey["1"].keys()) > 0:
            avg_overlaps.append(overlap[1])
            for q in questions:
                ratings[q].append(chat_survey["1"][q])

    correlations = dict((q, stats.pearsonr(avg_overlaps, ratings[q])) for q in questions)

    return correlations


def compute_statistics(args, lexicon, schema, scenario_db, transcripts, surveys, questions=("persuasive", "negotiator")):
    if not os.path.exists(os.path.dirname(args.stats_output)) and len(os.path.dirname(args.stats_output)) > 0:
        os.makedirs(os.path.dirname(args.stats_output))

    stats = {}
    statsfile = open(args.stats_output, 'w')
    # stats["total"] = total_stats = get_total_statistics(transcripts, scenario_db)
    stats = {"total":{}}
    stats["total"]["avg_description_overlap"] = avg_overlap = compute_avg_description_overlap(transcripts)
    # print "Aggregated total dataset statistics"
    # print_group_stats(total_stats)

    print "Avg. description overlap: %2.4f" % avg_overlap

    corr = get_overlap_correlation(transcripts, surveys, questions)
    print "Correlations between ratings and persuasivness:"
    for q in questions:
        print "%s" % q, corr[q]
    # Speech acts
    json.dump(stats, statsfile)
    statsfile.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--schema-path', help='Input path that describes the schema of the domain', required=True)
    parser.add_argument('--scenarios-path', help='Output path for the scenarios generated', required=True)
    parser.add_argument('--transcripts', required=True, help='Path to JSON file containing transcripts')
    parser.add_argument('--surveys', required=True, help='Path to JSON file containing surveys')
    parser.add_argument('--stats-output')
    args = parser.parse_args()
    schema = Schema(args.schema_path)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))
    transcripts = json.load(open(args.transcripts, 'r'))
    surveys = json.load(open(args.surveys, 'r'))[1]

    compute_statistics(args, None, schema, scenario_db, transcripts, surveys)
