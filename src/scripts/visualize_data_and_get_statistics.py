from src.basic.lexicon import Lexicon
from src.basic.util import read_json
from src.model.preprocess import Preprocessor

__author__ = 'anushabala'

import os
from argparse import ArgumentParser
from src.basic.scenario_db import ScenarioDB, add_scenario_arguments
from src.basic.schema import Schema
import json
from dataset_statistics import *


def get_html_for_transcript(chat):
    chat_html= ['<table>', '<tr>', '<td width=\"50%%\">']
    events = [Event.from_dict(e) for e in chat["events"]]

    if len(events) == 0:
        return False, None

    current_user = 0
    chat_html.append("<b>Agent 0</b></td><td width=\"50%%\"><b>Agent 1</b></td></tr><tr><td width=\"50%%\">")

    for event in events:
        if event.agent != current_user:
            chat_html.append('</td>')
            if current_user == 1:
                chat_html.append('</tr><tr>')
            chat_html.append('<td width=\"50%%\">')
        else:
            chat_html.append('<br>')

        current_user = event.agent
        if event.action == 'message':
            chat_html.append(event.data)
        elif event.action == 'select':
            chat_html.append("Selected " + ", ".join(event.data.values()))

    if current_user == 0:
        chat_html.append('</td><td width=\"50%%\">LEFT</td></tr>')

    chat_html.append('</table>')
    chat_html.append('<br>')
    completed = False if chat["outcome"] is None or chat["outcome"]["reward"] == 0 else True
    if completed:
        chat_html.insert(0, '<div style=\"color:#0000FF\">')
    else:
        chat_html.insert(0, '<div style=\"color:#FF0000\">')
    chat_html.append('</div>')
    chat_html.append('</div>')

    return completed, chat_html


def render_scenario(scenario):
    html = ["<div>"]
    for (idx, kb) in enumerate(scenario.kbs):
        kb_dict = kb.to_dict()
        attributes = [attr.name for attr in scenario.attributes]
        scenario_alphas = scenario.alphas
        if len(scenario_alphas) == 0:
            scenario_alphas = ['default' * len(scenario.attributes)]
        alphas = dict((attr.name, alpha) for (attr, alpha) in zip(scenario.attributes, scenario_alphas))
        html.append("<div style=\"display:inline-block;\"><table><tr>"
                    "<td colspan=\"%d\" style=\"color:#8012b7;\"><b>Agent %d</b></td></tr>" % (len(attributes), idx))

        for attr in attributes:
            html.append("<th><b>%s</b> (Alpha: %.3f)</th>" % (attr, alphas[attr]))
        html.append("</tr>")

        for item in kb_dict:
            html.append("<tr>")
            for attr in attributes:
                html.append("<td>%s</td>" % item[attr])
            html.append("</tr>")

        html.append("</table></div>")

    return html


def aggregate_chats(transcripts, scenario_db):
    html = ['<!DOCTYPE html>','<html>',
            '<head><style>table{ table-layout: fixed; width: 600px; border-collapse: collapse; } '
            'tr:nth-child(n) { border: solid thin;}</style></head><body>']
    completed_chats = []
    incomplete_chats = []
    total = 0
    num_completed = 0
    for (idx, chat) in enumerate(transcripts):
        completed, chat_html = get_html_for_transcript(chat)
        scenario_html = render_scenario(scenario_db.get(chat["scenario_uuid"]))
        if completed:
            num_completed += 1
            completed_chats.append('<div><h3>Chat %d</h3>' % idx)
            completed_chats.append('<h4>Scenario</h4>')
            completed_chats.extend(scenario_html)
            completed_chats.append('<h4>Dialogue</h4>')
            completed_chats.extend(chat_html)
            completed_chats.append('</div>')
            completed_chats.append("<hr>")
        else:
            if chat_html is not None:
                incomplete_chats.extend(chat_html)
                incomplete_chats.extend(scenario_html)
        total += 1

    html.extend(['<h3>Total number of chats: %d</h3>' % total,
                 '<h3>Number of chats completed: %d</h3>' % num_completed,
                 '<hr>'])
    html.extend(completed_chats)
    html.extend(incomplete_chats)
    html.append('</body></html>')
    return html


if __name__ == "__main__":
    parser = ArgumentParser()
    add_scenario_arguments(parser)
    parser.add_argument('--transcripts', type=str, default='transcripts.json', help='Path to directory containing transcripts')
    parser.add_argument('--html-output', type=str, required=True, help='Name of file to write HTML report to')
    parser.add_argument('--stats-output', type=str, required=True, help='Name of file to write JSON statistics to')
    parser.add_argument('--domain', type=str, choices=['MutualFriends', 'Matchmaking'])
    parser.add_argument('--alpha-stats', action='store_true', help='Get statistics grouped by alpha values')
    parser.add_argument('--item-stats', action='store_true',
                        help='Get statistics grouped by number of items in scenarios')
    parser.add_argument('--plot-item-stats', type=str, default=None,
                        help='If provided, and if --item-stats is specified, plots the relationship between # of items '
                             'and various stats to the provided path.')

    args = parser.parse_args()
    schema = Schema(args.schema_path, args.domain)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))
    transcripts = json.load(open(args.transcripts, 'r'))
    if not os.path.exists(os.path.dirname(args.html_output)) and len(os.path.dirname(args.html_output)) > 0:
        os.makedirs(os.path.dirname(args.html_output))
    if not os.path.exists(os.path.dirname(args.stats_output)) and len(os.path.dirname(args.stats_output)) > 0:
        os.makedirs(os.path.dirname(args.stats_output))

    html_lines = aggregate_chats(transcripts, scenario_db)

    outfile = open(args.html_output, 'w')
    for line in html_lines:
        outfile.write(line+"\n")
    outfile.close()

    stats = {}
    statsfile = open(args.stats_output, 'w')
    stats["total"] = total_stats = get_total_statistics(transcripts, scenario_db)
    print "Aggregated dataset statistics"
    print_group_stats(total_stats)

    if args.alpha_stats:
        print "-----------------------------------"
        print "-----------------------------------"
        print "Getting statistics grouped by alpha values...."
        stats["by_alpha"] = stats_by_alpha = get_statistics_by_alpha(transcripts, scenario_db)
        print_stats(stats_by_alpha, stats_type="alphas")
    if args.item_stats:
        print "-----------------------------------"
        print "-----------------------------------"
        print "Getting statistics grouped by number of items..."
        stats["by_num_items"] = stats_by_num_items = get_statistics_by_items(transcripts, scenario_db)
        print_stats(stats_by_num_items, stats_type="number of items")

        if args.plot_item_stats is not None:
            plot_num_items_stats(stats_by_num_items, args.plot_item_stats)

    lexicon = Lexicon(schema, False)
    preprocessor = Preprocessor(schema, lexicon, 'canonical', 'canonical', 'canonical', False)
    strategy_stats = analyze_strategy(transcripts, scenario_db, preprocessor)
    print_strategy_stats(strategy_stats)
    stats["speech_act"] = strategy_stats['speech_act']
    stats["kb_strategy"] = strategy_stats['kb_strategy']

    json.dump(stats, statsfile)
    statsfile.close()
