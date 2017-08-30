__author__ = 'anushabala'

from argparse import ArgumentParser
from cocoa.core.negotiation.tokenizer import tokenize
import json
import os
from scipy import stats as scipy_stats
import tf_idf
import ngram
import utils


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
    description = set()
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

    return total_overlap / num_agents


def get_overlap_correlation(transcripts, surveys, questions=("persuasive", "negotiator")):
    avg_overlaps = []
    ratings = dict((q, []) for q in questions)
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


def compute.core.statistics(transcripts, stats, surveyed_chats):
    total_turns = {"total": 0.}
    total_tokens = {"total": 0.}
    total_chats = {"total": 0.}
    total_time = {"total": 0.}
    for t in transcripts:
        if t["uuid"] not in surveyed_chats:
            continue
        turns = utils.get_turns_per_agent(t)
        tokens = utils.get_total_tokens_per_agent(t)
        time = utils.get_avg_time_taken(t)

        # Note: this check is redundant now because we already filter for chats that have surveys, and only chats
        # that are complete / partial can be submitted with surveys. This check is just here to be compatible with
        # previous batches where the interface allowed submissions of incomplete/partial chats
        # 06/01/2017 -- disabled this check because the rejection criteria have changed - by default just look at all chats with surveys
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
            total_time[name] = 0.

        total_chats["total"] += 1
        total_turns["total"] += turns[0] + turns[1]
        total_tokens["total"] += tokens[0] + tokens[1]
        total_time["total"] += time

        total_chats[name] += 1
        total_turns[name] += turns[0] + turns[1]
        total_tokens[name] += tokens[0] + tokens[1]
        total_time[name] += time

    for key in total_chats.keys():
        stats["turns"][key] = total_turns[key] / total_chats[key]
        stats["tokens"][key] = total_tokens[key] / (total_chats[key] * 2)
        stats["num_completed"][key] = total_chats[key]
        stats["time"][key] = total_time[key] / total_chats[key]


def pretty_print_stats(stats, label):
    print "{:s}".format(label)
    for key in sorted(stats.keys()):
        print "\t{key: <20}: {val:2.3f}".format(key=key, val=stats[key])


def get_statistics(transcripts, survey_data, questions=("persuasive", "negotiator")):
    surveyed_chats = survey_data[0].keys()
    surveys = survey_data[1]

    stats_out_path = os.path.join(stats_output, "stats.json")
    statsfile = open(stats_out_path, 'w')
    stats = {
        "avg_description_overlap": {},
        "turns": {},
        "tokens": {},
        "num_completed": {},
        "time": {}
    }
    # stats["avg_description_overlap"] = avg_overlap = compute_avg_description_overlap(transcripts, surveyed_chats)
    compute.core.statistics(transcripts, stats, surveyed_chats)

    # print "Avg. description overlap: %2.4f" % avg_overlap

    # corr = get_overlap_correlation(transcripts, surveys, questions)
    # print "Correlations between ratings and persuasivness:"
    # for q in questions:
    #     print "%s" % q, corr[q]

    json.dump(stats, statsfile)

    pretty_print_stats(stats["turns"], "Average # of turns")
    pretty_print_stats(stats["tokens"], "Average # of tokens per agent")
    pretty_print_stats(stats["time"], "Average time taken")
    pretty_print_stats(stats["num_completed"], "Number of chats")

    statsfile.close()


def analyze_tf_idf(transcripts, grouping_fn=tf_idf.group_by_category_and_role, n=1, output_dir=None):
    if output_dir is None:
        output_dir = stats_output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    top_features_by_agent = []
    for agent_type in args.agent_types:
        grouped_chats = grouping_fn(transcripts, agent_type)
        tfidf = tf_idf.TfIdfCalculator(grouped_chats, top_n=20, agent_type=agent_type, ngrams=n)
        top_features_by_cat = tfidf.analyze()
        top_features_by_agent.append(top_features_by_cat)
        tfidf.plot_score_distribution(output_dir, suffix="_{:d}grams".format(n))

    tf_idf.plot_top_tokens(top_features_by_agent,
                           agents=args.agent_types,
                           output_dir=output_dir,
                           suffix="_{:d}grams".format(n))
    tf_idf.write_to_file(top_features_by_agent,
                         agents=args.agent_types,
                         output_dir=output_dir,
                         suffix="_{:d}grams".format(n))


def analyze_ngrams(transcripts, grouping_fn=ngram.group_text_by_category, output_dir=None, n=5):
    if output_dir is None:
        output_dir = stats_output

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    top_ngrams_by_agent = []
    for agent_type in args.agent_types:
        grouped_utterances = grouping_fn(transcripts, agent_type)
        analyzer = ngram.NgramAnalyzer(grouped_utterances, n=n, agent_type=agent_type)

        top_ngrams_by_cat = analyzer.analyze()
        top_ngrams_by_agent.append(top_ngrams_by_cat)

    ngram.plot_top_ngrams(top_ngrams_by_agent, agents=args.agent_types,
                          output_dir=output_dir,
                          suffix="_{:d}grams".format(n))
    ngram.write_to_file(top_ngrams_by_agent, agents=args.agent_types,
                        output_dir=output_dir,
                        suffix="_{:d}grams".format(n))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output-dir', required=True, help='Directory containing all output from website')
    parser.add_argument('--agent-types', nargs='+', default=['human'], help='Types of agents to get statistics for')
    parser.add_argument('--limit', type=int, default=-1, help='Analyze the first N transcripts')
    parser.add_argument('--tf-idf', action='store_true', help='Whether to perform tf-idf analysis or not')
    parser.add_argument('--ngram', action='store_true', help='Whether to perform ngram analysis or not')
    args = parser.parse_args()
    out_dir = args.output_dir
    if not os.path.exists(out_dir):
        raise ValueError("Output directory {:s} doesn't exist".format(out_dir))

    transcripts = json.load(open(os.path.join(out_dir, "transcripts", "transcripts.json"), 'r'))
    if args.limit > 0:
        transcripts = transcripts[:args.limit]
    survey_data = json.load(open(os.path.join(out_dir, "transcripts", "surveys.json"), 'r'))

    stats_output = os.path.join(out_dir, "stats")
    print stats_output
    if not os.path.exists(stats_output):
        os.makedirs(stats_output)

    get_statistics(transcripts, survey_data)

    if args.tf_idf:
        # tf_idf_by_category(transcripts)
        # tf_idf_by_role(transcripts)
        # tf_idf_by_winner(transcripts)
        tf_idf_dir = os.path.join(stats_output, 'tfidf')
        if not os.path.exists(tf_idf_dir):
            os.makedirs(tf_idf_dir)
        n_range = xrange(2, 4)
        for i in n_range:
            analyze_tf_idf(transcripts,
                           grouping_fn=tf_idf.group_by_category_role_winner,
                           n=i,
                           output_dir=os.path.join(tf_idf_dir, 'by_winner'))

    if args.ngram:
        ngram_dir = os.path.join(stats_output, 'ngram')
        if not os.path.exists(ngram_dir):
            os.makedirs(ngram_dir)
        analyze_ngrams(transcripts, grouping_fn=ngram.group_text_by_category,
                       output_dir=os.path.join(ngram_dir, 'by_category'))
        analyze_ngrams(transcripts, grouping_fn=ngram.group_text_by_role,
                       output_dir=os.path.join(ngram_dir, 'by_role'))
        analyze_ngrams(transcripts, grouping_fn=ngram.group_text_by_winner,
                       output_dir=os.path.join(ngram_dir, 'by_winner'))
