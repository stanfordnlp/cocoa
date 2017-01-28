__author__ = 'anushabala'
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import numpy as np
import json
from src.basic.systems.human_system import HumanSystem
import os


def visualize(question, responses_tuples, titles, save_path):
    N = 5
    ind = np.arange(N)  # the x locations for the groups
    width = 0.15       # the width of the bars

    fig, ax = plt.subplots()
    i = 0
    all_rects = []
    for (responses, color) in responses_tuples:
        rects = ax.bar(ind + i * width, responses, width, color=color)
        all_rects.append(rects)
        i += 1

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Percentage')
    ax.set_title('%s Percentages By Model' % question)
    ax.set_xticks(2*width + ind)
    ax.set_xticklabels(('Bad', 'Mediocre', 'Acceptable', 'Good', 'Excellent'))

    ax.legend([r[0] for r in all_rects], titles, loc="upper left")

    plt.savefig(save_path)


def get_complete_scenarios(surveys, transcripts, num_agent_types=4):
    scenario_ids = {t["scenario_uuid"] for t in transcripts}
    scenario_agent_mappings = dict((i, set()) for i in scenario_ids)
    surveyed_chats = surveys[0].keys()
    for chat in transcripts:
        if chat["uuid"] in surveyed_chats:
            agent_types = chat["agents"]
            sid = chat["scenario_uuid"]
            a = agent_types['0'] if agent_types['1'] == 'human' else agent_types['1']
            scenario_agent_mappings[sid].add(a)

    return {sid for (sid, types) in scenario_agent_mappings.items() if len(types) == num_agent_types}


def aggregate_responses(question, completed_scenarios, surveys, transcripts,
                        bots=['human', 'rulebased', 'static-neural', 'dynamic-neural'], n=5):
    agent_type_mappings = surveys[0]
    # print agent_type_mappings
    responses = surveys[1]

    question_responses = dict((b, dict((x, 0.0) for x in xrange(0, n))) for b in bots)

    for chat in transcripts:
        cid = chat["uuid"]
        if chat["scenario_uuid"] in completed_scenarios and cid in agent_type_mappings.keys():
            agents = agent_type_mappings[cid]
            agent_type = agents['0'] if agents['1'] == HumanSystem.name() else agents['1']
            rating = responses[cid][question]
            question_responses[agent_type][rating - 1] += 1.0

    return question_responses


def visualize_surveys(surveys, transcripts, output_dir,
                      questions=["fluent", "cooperative", "humanlike", "correct"],
                      bots=['human', 'rulebased', 'static-neural', 'dynamic-neural'],
                      colors=["r", "y", "b", "g"]):
    completed_scenarios = get_complete_scenarios(surveys, transcripts, len(bots))
    for q in questions:
        question_responses = aggregate_responses(q, completed_scenarios, surveys, transcripts, bots)
        responses_tuples = []
        for (agent_type, c) in zip(bots, colors):
            ratings = question_responses[agent_type]

            ratings = {k: v/sum(ratings.values()) for (k, v) in ratings.items()}
            sorted_ratings = [ratings[k] for k in sorted(ratings.keys())]
            responses_tuples.append((sorted_ratings, c))

        visualize(q, responses_tuples, bots, os.path.join(output_dir, '%s.png' % q))



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--transcripts', type=str, help='Path to transcripts JSON file')
    parser.add_argument('--surveys', type=str, help='Path to surveys JSON file')
    parser.add_argument('--output-dir', type=str, help='Path to directory to write files to')

    args = parser.parse_args()
    surveys_json = json.load(open(args.surveys, 'r'))
    transcripts_json = json.load(open(args.transcripts, 'r'))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    visualize_surveys(surveys_json, transcripts_json, args.output_dir)



