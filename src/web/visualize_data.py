from src.basic.util import read_json

__author__ = 'anushabala'

import os
from argparse import ArgumentParser
from src.basic.scenario_db import ScenarioDB, add_scenario_arguments
from src.basic.schema import Schema
from src.basic.event import Event
import json


def get_html_for_transcript(i, chat):
    chat_html= ['<div><h4>Chat %d</h4>' % i, '<table>', '<tr>', '<td width=\"50%%\">']
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


def render_scenario(schema, scenario):
    html = ["<div>"]
    for (idx, kb) in enumerate(scenario.kbs):
        kb_dict = kb.to_dict()
        attributes = [attr.name for attr in schema.attributes]
        html.append("<div style=\"display:inline-block;\"><table><tr>"
                    "<td colspan=\"%d\" style=\"color:#8012b7;\"><b>Agent %d</b></td></tr>" % (len(attributes), idx))

        for attr in attributes:
            html.append("<th><b>%s</b></th>" % attr)
        html.append("</tr>")

        for item in kb_dict:
            html.append("<tr>")
            for attr in attributes:
                html.append("<td>%s</td>" % item[attr])
            html.append("</tr>")

        html.append("</table></div>")

    html.append("</div>")
    return html


def aggregate_chats(transcripts, scenario_db, schema):
    html = ['<!DOCTYPE html>','<html>',
            '<head><style>table{ table-layout: fixed; width: 600px; border-collapse: collapse; } '
            'tr:nth-child(n) { border: solid thin;}</style></head><body>']
    completed_chats = []
    incomplete_chats = []
    total = 0
    num_completed = 0
    for (idx, chat) in enumerate(transcripts):
        completed, chat_html = get_html_for_transcript(idx, chat)
        scenario_html = render_scenario(schema, scenario_db.get(chat["scenario_uuid"]))
        if completed:
            num_completed += 1
            completed_chats.extend(chat_html)
            completed_chats.extend(scenario_html)
        else:
            if chat_html is not None:
                incomplete_chats.extend(chat_html)
                incomplete_chats.extend(scenario_html)
        total += 1

    html.extend(['<h3>Total number of chats: %d</h3>' % total,
                 '<h3>Number of chats completed: %d</h3>' % num_completed])
    html.extend(completed_chats)
    html.extend(incomplete_chats)
    html.append('</body></html>')
    return html

if __name__ == "__main__":
    parser = ArgumentParser()
    add_scenario_arguments(parser)
    parser.add_argument('--transcripts', type=str, default='transcripts.json', help='Path to directory containing transcripts')
    parser.add_argument('--output', type=str, required=True, help='Name of file to write report to')
    parser.add_argument('--domain', type=str, choices=['MutualFriends', 'Matchmaking'])

    args = parser.parse_args()
    schema = Schema(args.schema_path, args.domain)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))
    transcripts = json.load(open(args.transcripts, 'r'))
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    html_lines = aggregate_chats(transcripts, scenario_db, schema)

    outfile = open(args.output, 'w')
    for line in html_lines:
        outfile.write(line+"\n")
    outfile.close()
