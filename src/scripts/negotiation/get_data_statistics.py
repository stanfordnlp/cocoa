__author__ = 'anushabala'

from src.turk.accept_negotiation_hits import reject_transcript, get_turns_per_agent, get_total_tokens_per_agent
from argparse import ArgumentParser
from src.model.preprocess import tokenize
import json
import os
from scipy import stats as scipy_stats
from tf_idf import TfIdfCalculator, plot_top_tokens


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

    correlations = dict((q, scipy_stats.pearsonr(avg_overlaps, ratings[q])) for q in questions)

    return correlations


def compute_basic_statistics(transcripts, stats, surveyed_chats):
    total_turns = {"total":0.}
    total_tokens = {"total": 0.}
    total_chats = {"total": 0.}
    for t in transcripts:
        if t["uuid"] not in surveyed_chats:
            continue
        turns = get_turns_per_agent(t)
        tokens = get_total_tokens_per_agent(t)

        # Note: this check is redundant now because we already filter for chats that have surveys, and only chats
        # that are complete / partial can be submitted with surveys. This check is just here to be compatible with
        # previous batches where the interface allowed submissions of incomplete/partial chats
        # 06/01/2017 -- disabled this check because the rejection criteria have changed
        # if reject_transcript(t):
        #     continue

        scenario = t["scenario"]
        category = scenario["category"]
        agent_types = t["agents"]
        agent = "human"
        if agent_types["0"] != agent:
            agent = agent_types["0"]
        elif agent_types["1"] != agent:
            agent = agent_types["1"]

        name = "{:s}_{:s}".format(agent, category)
        if name not in total_turns:
            total_turns[name] = 0.
            total_chats[name] = 0.
            total_tokens[name] = 0.

        total_chats["total"] += 1
        total_turns["total"] += turns[0] + turns[1]
        total_tokens["total"] += tokens[0] + tokens[1]

        total_chats[name] += 1
        total_turns[name] += turns[0] + turns[1]
        total_tokens[name] += tokens[0] + tokens[1]

    for key in total_chats.keys():
        stats["turns"][key] = total_turns[key]/total_chats[key]
        stats["tokens"][key] = total_tokens[key]/(total_chats[key]*2)
        stats["num_completed"][key] = total_chats[key]


def pretty_print_stats(stats, label):
    print "Grouped statistics for: {:s} ".format(label)
    for key in sorted(stats.keys()):
        print "\t{key: <20}: {val:2.3f}".format(key=key, val=stats[key])


def get_statistics(args, transcripts, survey_data, questions=("persuasive", "negotiator")):
    if not os.path.exists(os.path.dirname(args.stats_output)) and len(os.path.dirname(args.stats_output)) > 0:
        os.makedirs(os.path.dirname(args.stats_output))

    surveyed_chats = survey_data[0].keys()
    surveys = survey_data[1]

    stats_out_path = os.path.join(args.stats_output, "stats.json")
    statsfile = open(stats_out_path, 'w')
    stats = {
        "avg_description_overlap":{},
        "turns": {},
        "tokens": {},
        "num_completed": {}
    }
    stats["avg_description_overlap"] = avg_overlap = compute_avg_description_overlap(transcripts, surveyed_chats)
    # print "Aggregated total dataset statistics"
    # print_group_stats(total_stats)
    compute_basic_statistics(transcripts, stats, surveyed_chats)

    print "Avg. description overlap: %2.4f" % avg_overlap

    corr = get_overlap_correlation(transcripts, surveys, questions)
    print "Correlations between ratings and persuasivness:"
    for q in questions:
        print "%s" % q, corr[q]
    # Speech acts
    json.dump(stats, statsfile)

    pretty_print_stats(stats["turns"], "Average # of turns")
    pretty_print_stats(stats["tokens"], "Average # of tokens")
    pretty_print_stats(stats["num_completed"], "Number of chats")

    statsfile.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--schema-path', help='Input path that describes the schema of the domain', required=True)
    parser.add_argument('--scenarios-path', help='Output path for the scenarios generated', required=True)
    parser.add_argument('--transcripts', required=True, help='Path to JSON file containing transcripts')
    parser.add_argument('--surveys', required=True, help='Path to JSON file containing surveys')
    parser.add_argument('--stats-output', required=True, help='Directory to output stats to')
    parser.add_argument('--agent-types', nargs='+', default=['human'], help='Types of agents to get statistics for')
    args = parser.parse_args()
    transcripts = json.load(open(args.transcripts, 'r'))
    survey_data = json.load(open(args.surveys, 'r'))

    get_statistics(args, transcripts, survey_data)

    top_features_by_agent = []
    for agent_type in args.agent_types:
        tfidf = TfIdfCalculator(transcripts, top_n=20, agent_type=agent_type)
        top_features_by_cat = tfidf.analyze()
        top_features_by_agent.append(top_features_by_cat)

    plot_top_tokens(top_features_by_agent, agents=args.agent_types, output_dir=args.stats_output)
