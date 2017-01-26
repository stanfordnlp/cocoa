from src.basic.util import read_json

__author__ = 'anushabala'

import os
from argparse import ArgumentParser
from src.basic.scenario_db import ScenarioDB, add_scenario_arguments
from src.basic.schema import Schema
from src.basic.event import Event
import json


def add_visualization_arguments(parser):
    parser.add_argument('--html-output', type=str, required=True, help='Name of file to write HTML report to')


def get_html_for_transcript(chat, agent=None, partner_type='Human'):
    chat_html= ['<table>', '<tr>', '<td width=\"50%%\">']
    events = [Event.from_dict(e) for e in chat["events"]]

    if len(events) == 0:
        return False, None

    if agent == 0:
        chat_html.append("<b>Agent 0 (You)</b></td><td width=\"50%%\"><b>Agent 1 (Partner: %s)</b></td></tr><tr><td width=\"50%%\">" % partner_type)
    elif agent == 1:
        chat_html.append("<b>Agent 0 (Partner: %s)</b></td><td width=\"50%%\"><b>Agent 1 (You)</b></td></tr><tr><td width=\"50%%\">" % partner_type)
    else:
        chat_html.append("<b>Agent 0 (%s) </b></td><td width=\"50%%\"><b>Agent 1 (%s) </b></td></tr><tr><td width=\"50%%\">" % (chat['agents']['0'], chat['agents']['1']))

    current_user = 0

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


def visualize_chat(chat, scenario_db, agent=None, partner_type='Human'):
    completed, chat_html = get_html_for_transcript(chat, agent, partner_type)
    if chat_html is None:
        return False, None

    scenario_html = render_scenario(scenario_db.get(chat["scenario_uuid"]))

    html_lines = []
    html_lines.append('<h4>Scenario</h4>')
    html_lines.extend(scenario_html)
    html_lines.append('<h4>Dialogue</h4>')
    html_lines.extend(chat_html)

    return completed, html_lines


def aggregate_chats(transcripts, scenario_db):
    html = ['<!DOCTYPE html>','<html>',
            '<head><style>table{ table-layout: fixed; width: 600px; border-collapse: collapse; } '
            'tr:nth-child(n) { border: solid thin;}</style></head><body>']
    completed_chats = []
    incomplete_chats = []
    total = 0
    num_completed = 0
    for (idx, chat) in enumerate(transcripts):
        completed, chat_html = visualize_chat(chat, scenario_db)
        if completed:
            num_completed += 1
            completed_chats.extend(chat_html)
            completed_chats.append('</div>')
            completed_chats.append("<hr>")
        else:
            if chat_html is not None:
                incomplete_chats.extend(chat_html)
                incomplete_chats.append('</div>')
                incomplete_chats.append("<hr>")
        total += 1

    html.extend(['<h3>Total number of chats: %d</h3>' % total,
                 '<h3>Number of chats completed: %d</h3>' % num_completed,
                 '<hr>'])
    html.extend(completed_chats)
    html.extend(incomplete_chats)
    html.append('</body></html>')
    return html


def visualize_transcripts(args, scenario_db, transcripts):
    if not os.path.exists(os.path.dirname(args.html_output)) and len(os.path.dirname(args.html_output)) > 0:
        os.makedirs(os.path.dirname(args.html_output))

    html_lines = aggregate_chats(transcripts, scenario_db)

    outfile = open(args.html_output, 'w')
    for line in html_lines:
        outfile.write(line+"\n")
    outfile.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    add_scenario_arguments(parser)
    add_visualization_arguments(parser)
    parser.add_argument('--transcripts', type=str, default='transcripts.json', help='Path to directory containing transcripts')
    parser.add_argument('--tagged-data', action='store_true',
                        help='If true, script assumes that the dataset to visualize is tagged, so event metadata '
                             'rather than raw event data is visualized')
    args = parser.parse_args()
    schema = Schema(args.schema_path)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))
    transcripts = json.load(open(args.transcripts, 'r'))

    visualize_transcripts(args, scenario_db, transcripts)
