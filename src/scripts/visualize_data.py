from src.basic.util import read_json

__author__ = 'anushabala'

import os
from argparse import ArgumentParser
from src.basic.scenario_db import ScenarioDB, Scenario, add_scenario_arguments
from src.basic.schema import Schema
from src.basic.event import Event
from src.basic.util import write_json, read_json
import numpy as np
import json
import datetime
from itertools import izip
from collections import defaultdict

def add_visualization_arguments(parser):
    parser.add_argument('--html-output', help='Name of directory to write HTML report to')
    parser.add_argument('--viewer-mode', action='store_true', help='Output viewer instead of single html')
    parser.add_argument('--css-file', default='chat_viewer/css/my.css', help='css for tables/scenarios and chat logs')

#questions = ['fluent', 'fluent_text', 'correct', 'correct_text', 'cooperative', 'cooperative_text', 'strategic', 'strategic_text', 'humanlike', 'humanlike_text', 'comments']
QUESTIONS = ['fluent', 'correct', 'cooperative', 'humanlike']

# Canonical names to be displayed
AGENT_NAMES = {'human': 'human', 'rulebased': 'rule-based', 'dynamic-neural': 'DynoNet', 'static-neural': 'StanoNet'}

def get_scenario(chat):
    scenario = Scenario.from_dict(None, chat['scenario'])
    return scenario

def render_chat(chat, agent=None, partner_type='human'):
    events = [Event.from_dict(e) for e in chat["events"]]

    if len(events) == 0:
        return False, None

    chat_html= ['<div class=\"chatLog\">',
            '<div class=\"divTitle\"> Chat Log </div>',
            '<table class=\"chat\">']
    agent_str = {0: '', 1: ''}

    # Used for visualizing chat during debugging
    if agent is not None:
        agent_str[agent] = 'Agent %d (you)' % agent
        agent_str[1 - agent] = 'Agent %d (%s)' % (1 - agent, partner_type)
    elif 'agents' in chat and chat['agents']:
        for agent in (0, 1):
            agent_str[agent] = 'Agent %d (%s)' % (agent, AGENT_NAMES[chat['agents'][str(agent)]])
    else:
        for agent in (0, 1):
            agent_str[agent] = 'Agent %d (%s)' % (agent, 'unknown')

    for event in events:
        t = datetime.datetime.fromtimestamp(float(event.time)).strftime('%Y-%m-%d %H:%M:%S')
        a = agent_str[event.agent]
        if event.action == 'message':
            s = event.data
        elif event.action == 'select':
            s = 'SELECT (' + ' || '.join(event.data.values()) + ')'
        row = '<tr class=\"agent%d\">\
                <td class=\"time\">%s</td>\
                <td class=\"agent\">%s</td>\
                <td class=\"message\">%s</td>\
               </tr>' % (event.agent, t, a, s)
        chat_html.append(row)

    chat_html.extend(['</table>', '</div>'])

    completed = False if chat["outcome"] is None or chat["outcome"]["reward"] == 0 else True
    return completed, chat_html

    #if agent == 0:
    #    chat_html.append("<b>Agent 0 (You)</b></td><td width=\"50%%\"><b>Agent 1 (Partner: %s)</b></td></tr><tr><td width=\"50%%\">" % partner_type)
    #elif agent == 1:
    #    chat_html.append("<b>Agent 0 (Partner: %s)</b></td><td width=\"50%%\"><b>Agent 1 (You)</b></td></tr><tr><td width=\"50%%\">" % partner_type)
    #elif 'agents' in chat:
    #    chat_html.append("<b>Agent 0 (%s) </b></td><td width=\"50%%\"><b>Agent 1 (%s) </b></td></tr><tr><td width=\"50%%\">" % (chat['agents']['0'], chat['agents']['1']))
    #else:
    #    chat_html.append("<b>Agent 0 </b></td><td width=\"50%%\"><b>Agent 1 </b></td></tr><tr><td width=\"50%%\">")

    #current_user = 0

    #for event in events:
    #    if event.agent != current_user:
    #        chat_html.append('</td>')
    #        if current_user == 1:
    #            chat_html.append('</tr><tr>')
    #        chat_html.append('<td width=\"50%%\">')
    #    else:
    #        chat_html.append('<br>')

    #    current_user = event.agent
    #    if event.action == 'message':
    #        chat_html.append(event.data)
    #    elif event.action == 'select':
    #        chat_html.append("Selected " + ", ".join(event.data.values()))

    #if current_user == 0:
    #    chat_html.append('</td><td width=\"50%%\">LEFT</td></tr>')

    #chat_html.append('</table>')
    #chat_html.append('<br>')
    #completed = False if chat["outcome"] is None or chat["outcome"]["reward"] == 0 else True
    #if completed:
    #    chat_html.insert(0, '<div style=\"color:#0000FF\">')
    #else:
    #    chat_html.insert(0, '<div style=\"color:#FF0000\">')
    #chat_html.append('</div>')
    #chat_html.append('</div>')

    #return completed, chat_html

def _render_response(response, agent_id, agent):
    html = []
    html.append('<table class=\"response%d\">' % agent_id)
    html.append('<tr><td colspan=\"4\" class=\"agentLabel\">Response to agent %d (%s)</td></tr>' % (agent_id, AGENT_NAMES[agent]))
    html.append('<tr>%s</tr>' % (''.join(['<th>%s</th>' % x for x in ('Question', 'Mean', 'Response', 'Justification')])))
    for question in QUESTIONS:
        if question not in response: #or question == 'comments' or question.endswith('text'):
            continue
        else:
            scores = response[question]
            if question+'_text' in response:
                just = response[question+'_text']
                assert len(scores) == len(just)
            else:
                just = None

        if just is not None:
            n = len(scores)
            for i, (s, j) in enumerate(izip(scores, just)):
                html.append('<tr>')
                if i == 0:
                    html.append('<td rowspan=\"%d\">%s</td>' % (n, question))
                    html.append('<td rowspan=\"%d\">%s</td>' % (n, np.mean(scores)))
                html.append('<td>%d</td><td>%s</td>' % (s, j))
                html.append('</tr>')
        else:
            html.append('<tr>%s</tr>' % (''.join(['<td>%s</td>' % x for x in (question, np.mean(scores), ' / '.join([str(x) for x in scores]))])))

    if 'comments' in response:
        comment_str = response['comments'][0]
        if len(comment_str) > 0:
            html.append('<tr><td>%s</td><td colspan=3>%s</td></tr>' % ('comments', comment_str))

    html.append('</table>')
    return html

def render_scenario(scenario):
    html = ["<div class=\"scenario\">"]
    html.append('<div class=\"divTitle\">Scenario %s</div>' % scenario.uuid)
    for (idx, kb) in enumerate(scenario.kbs):
        kb_dict = kb.to_dict()
        attributes = [attr.name for attr in scenario.attributes]
        scenario_alphas = scenario.alphas
        if len(scenario_alphas) == 0:
            scenario_alphas = ['default' * len(scenario.attributes)]
        alphas = dict((attr.name, alpha) for (attr, alpha) in zip(scenario.attributes, scenario_alphas))
        html.append("<div class=\"kb%d\"><table><tr>"
                    "<td colspan=\"%d\" class=\"agentLabel\">Agent %d</td></tr>" % (idx, len(attributes), idx))

        for attr in attributes:
            html.append("<th>%s (%.1f)</th>" % (attr, alphas[attr]))
        html.append("</tr>")

        for item in kb_dict:
            html.append("<tr>")
            for attr in attributes:
                html.append("<td>%s</td>" % item[attr])
            html.append("</tr>")

        html.append("</table></div>")

    html.append("</div>")
    return html

def render_response(responses, agent_dict):
    html_lines = ["<div class=\"survey\">"]
    html_lines.append('<div class=\"divTitle\">Survey</div>')
    for agent_id, response in responses.iteritems():
        html_lines.append('<div class=\"response\">')
        response_html = _render_response(response, int(agent_id), agent_dict[agent_id])
        html_lines.extend(response_html)
        html_lines.append("</div>")
    html_lines.append("</div>")
    return html_lines

def visualize_chat(chat, agent=None, partner_type='Human', responses=None, id_=None):
    completed, chat_html = render_chat(chat, agent, partner_type)
    if chat_html is None:
        return False, None

    html_lines = []

    scenario_html = render_scenario(get_scenario(chat))
    html_lines.extend(scenario_html)

    html_lines.extend(chat_html)

    if responses:
        dialogue_id = chat['uuid']
        agents = chat['agents']
        response_html = render_response(responses[dialogue_id], agents)
        html_lines.extend(response_html)

    return completed, html_lines


def aggregate_chats(transcripts, responses=None, css_file=None):
    html = ['<!DOCTYPE html>','<html>',
            '<head><style>table{ table-layout: fixed; width: 600px; border-collapse: collapse; } '
            'tr:nth-child(n) { border: solid thin;}</style></head><body>']

    # inline css
    if css_file:
        html.append('<style>')
        with open(css_file, 'r') as fin:
            for line in fin:
                html.append(line.strip())
        html.append('</style>')

    completed_chats = []
    incomplete_chats = []
    total = 0
    num_completed = 0
    for (idx, chat) in enumerate(transcripts):
        completed, chat_html = visualize_chat(chat, responses=responses, id_=idx)
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


def visualize_transcripts(html_output, transcripts, responses=None, css_file=None):
    if not os.path.exists(os.path.dirname(html_output)) and len(os.path.dirname(html_output)) > 0:
        os.makedirs(os.path.dirname(html_output))

    html_lines = aggregate_chats(transcripts, responses, css_file)

    outfile = open(html_output, 'w')
    for line in html_lines:
        outfile.write(line+"\n")
    outfile.close()


def write_chat_htmls(transcripts, outdir, responses=None):
    outdir = os.path.join(outdir, 'chat_htmls')
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    for chat in transcripts:
        dialogue_id = chat['uuid']
        _, chat_html = visualize_chat(chat, responses=responses)
        if not chat_html:
            continue
        with open(os.path.join(outdir, dialogue_id+'.html'), 'w') as fout:
            # For debugging: write complete html file
            #fout.write("<!DOCTYPE html>\
            #        <html>\
            #        <head>\
            #        <link rel=\"stylesheet\" type\"text/css\" href=\"../css/my.css\"\
            #        </head>")
            fout.write('\n'.join(chat_html).encode('utf-8'))
            #fout.write("</html>")

def write_metadata(transcripts, outdir, responses=None):
    metadata = {'data': []}
    for chat in transcripts:
        if len(chat['events']) == 0:
            continue
        row = {}
        row['dialogue_id'] = chat['uuid']
        row['scenario_id'] = chat['scenario_uuid']
        scenario = get_scenario(chat)
        row['num_items'] = len(scenario.kbs[0].items)
        row['num_attrs'] = len(scenario.attributes)
        row['outcome'] = 'fail' if chat['outcome']['reward'] == 0 else 'success'
        row['agent0'] = AGENT_NAMES[chat['agents']['0']]
        row['agent1'] = AGENT_NAMES[chat['agents']['1']]
        if responses:
            dialogue_response = responses[chat['uuid']]
            question_scores = defaultdict(list)
            for agent_id, scores in dialogue_response.iteritems():
                for question in QUESTIONS:
                    question_scores[question].extend(scores[question])
            for question, scores in question_scores.iteritems():
                row[question] = np.mean(scores)
        metadata['data'].append(row)
    write_json(metadata, os.path.join(outdir, 'metadata.json'))

def write_viewer_data(html_output, transcripts, responses=None):
    if not os.path.exists(html_output):
        os.makedirs(html_output)
    write_metadata(transcripts, html_output, responses)
    write_chat_htmls(transcripts, html_output, responses)


if __name__ == "__main__":
    parser = ArgumentParser()
    add_scenario_arguments(parser)
    add_visualization_arguments(parser)
    parser.add_argument('--transcripts', type=str, default='transcripts.json', help='Path to directory containing transcripts')


    args = parser.parse_args()
    schema = Schema(args.schema_path)
    #scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))
    transcripts = read_json(args.transcripts)
    html_output = args.html_output

    if args.viewer_mode:
        # External js and css
        write_viewer_data(html_output, transcripts)
    else:
        # Inline style
        visualize_transcripts(html_output, transcripts, css_file=args.css_file)
