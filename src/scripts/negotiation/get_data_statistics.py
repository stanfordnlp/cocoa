__author__ = 'anushabala'

from src.turk.accept_negotiation_hits import is_chat_valid, is_partial_chat
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


def compute_avg_description_overlap(transcripts, surveyed_chats):
    num_agents = 0
    total_overlap = 0.
    for t in transcripts:
        if t["uuid"] not in surveyed_chats:
            continue
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


# todo copied these from accept_negotiation_hits.py--- consolidate
def get_turns_per_agent(transcript):
    turns = {0: 0, 1: 0}
    for event in transcript["events"]:
        if event["action"] == "message":
            turns[event["agent"]] += 1

    return turns


def get_avg_tokens_per_agent(transcript):
    tokens = {0: 0., 1: 0.}
    utterances = {0: 0., 1: 0.}
    for event in transcript["events"]:
        if event["action"] == "message":
            msg_tokens = tokenize(event["data"])
            tokens[event["agent"]] += len(msg_tokens)
            utterances[event["agent"]] += 1

    if utterances[0] != 0:
        tokens[0] /= utterances[0]
    if utterances[1] != 0:
        tokens[1] /= utterances[1]

    return tokens


def get_overall_statistics(transcripts, stats, surveyed_chats):
    total_turns = 0.
    avg_tokens = 0.
    total_agents = 0.
    for t in transcripts:
        if t["uuid"] not in surveyed_chats:
            continue

        # Note: this check is redundant now because we already filter for chats that have surveys, and only chats
        # that are complete / partial can be submitted with surveys. This check is just here to be compatible with
        # previous batches where the interface allowed submissions of incomplete/partial chats
        if (not is_chat_valid(t, 0) and not is_partial_chat(t, 0)) \
                or (not is_chat_valid(t, 1) and not is_partial_chat(t, 1)):
            continue
        turns = get_turns_per_agent(t)
        tokens = get_avg_tokens_per_agent(t)

        total_agents += 2
        total_turns += turns[0] + turns[1]
        avg_tokens += tokens[0] + tokens[1]

    print "Total chats: ", total_agents/2
    stats["avg_turns"] = total_turns / total_agents
    stats["avg_tokens"] = avg_tokens / total_agents

def compute_statistics(args, lexicon, schema, scenario_db, transcripts, survey_data, questions=("persuasive", "negotiator")):
    if not os.path.exists(os.path.dirname(args.stats_output)) and len(os.path.dirname(args.stats_output)) > 0:
        os.makedirs(os.path.dirname(args.stats_output))

    surveyed_chats = survey_data[0].keys()
    surveys = survey_data[1]

    stats = {}
    statsfile = open(args.stats_output, 'w')
    # stats["total"] = total_stats = get_total_statistics(transcripts, scenario_db)
    stats = {"total":{}}
    stats["total"]["avg_description_overlap"] = avg_overlap = compute_avg_description_overlap(transcripts, surveyed_chats)
    # print "Aggregated total dataset statistics"
    # print_group_stats(total_stats)
    get_overall_statistics(transcripts, stats["total"], surveyed_chats)

    print "Avg. description overlap: %2.4f" % avg_overlap

    corr = get_overlap_correlation(transcripts, surveys, questions)
    print "Correlations between ratings and persuasivness:"
    for q in questions:
        print "%s" % q, corr[q]
    # Speech acts
    json.dump(stats, statsfile)
    print stats
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
    survey_data = json.load(open(args.surveys, 'r'))

    compute_statistics(args, None, schema, scenario_db, transcripts, survey_data)
