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
import src.config as config

def add_html_visualizer_arguments(parser):
    parser.add_argument('--html-output', help='Name of directory to write HTML report to')
    parser.add_argument('--viewer-mode', action='store_true', help='Output viewer instead of single html')
    parser.add_argument('--css-file', default='chat_viewer/css/my.css', help='css for tables/scenarios and chat logs')

class HTMLVisualizer(object):
    @staticmethod
    def get_html_visualizer(*args):
        if config.task == config.MutualFriends:
            return MutualFriendsHTMLVisualizer(*args)
        elif config.task == config.Negotiation:
            return NegotiationHTMLVisualizer(*args)
        else:
            raise ValueError('Unknown task: %s.' % config.task)

class BaseHTMLVisualizer(object):
    agent_labels = None
    questions = None

    @classmethod
    def get_scenario(cls, chat):
        scenario = Scenario.from_dict(None, chat['scenario'])
        return scenario

    @classmethod
    def render_event(cls, event):
        if event.action == 'message':
            return event.data
        return None

    @classmethod
    def render_chat(cls, chat, agent=None, partner_type='human'):
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
                agent_str[agent] = 'Agent %d (%s)' % (agent, cls.agent_labels[chat['agents'][str(agent)]])
        else:
            for agent in (0, 1):
                agent_str[agent] = 'Agent %d (%s)' % (agent, 'unknown')

        for event in events:
            t = datetime.datetime.fromtimestamp(float(event.time)).strftime('%Y-%m-%d %H:%M:%S')
            a = agent_str[event.agent]
            # TODO: factor render_event
            if event.action == 'message':
                s = event.data
            elif event.action == 'select':
                s = 'SELECT (' + ' || '.join(event.data.values()) + ')'
            elif event.action == 'offer':
                s = 'OFFER $%.1f' % float(event.data)
            else:
                continue
            row = '<tr class=\"agent%d\">\
                    <td class=\"time\">%s</td>\
                    <td class=\"agent\">%s</td>\
                    <td class=\"message\">%s</td>\
                   </tr>' % (event.agent, t, a, s)

            chat_html.append(row)

        chat_html.extend(['</table>', '</div>'])

        completed = False if chat["outcome"] is None or chat["outcome"]["reward"] == 0 else True
        return completed, chat_html

    @classmethod
    def _render_response(cls, response, agent_id, agent):
        html = []
        html.append('<table class=\"response%d\">' % agent_id)
        html.append('<tr><td colspan=\"4\" class=\"agentLabel\">Response to agent %d (%s)</td></tr>' % (agent_id, cls.agent_labels[agent]))
        html.append('<tr>%s</tr>' % (''.join(['<th>%s</th>' % x for x in ('Question', 'Mean', 'Response', 'Justification')])))
        for question in cls.questions:
            if question not in response:
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

    @classmethod
    def render_scenario(cls, scenario):
        raise NotImplementedError

    @classmethod
    def render_response(cls, responses, agent_dict):
        html_lines = ["<div class=\"survey\">"]
        html_lines.append('<div class=\"divTitle\">Survey</div>')
        for agent_id, response in responses.iteritems():
            html_lines.append('<div class=\"response\">')
            response_html = cls._render_response(response, int(agent_id), agent_dict[agent_id])
            html_lines.extend(response_html)
            html_lines.append("</div>")
        html_lines.append("</div>")
        return html_lines

    @classmethod
    def visualize_chat(cls, chat, agent=None, partner_type='Human', responses=None, id_=None):
        completed, chat_html = cls.render_chat(chat, agent, partner_type)
        if chat_html is None:
            return False, None

        html_lines = []

        scenario_html = cls.render_scenario(cls.get_scenario(chat))
        html_lines.extend(scenario_html)

        html_lines.extend(chat_html)

        if responses:
            dialogue_id = chat['uuid']
            agents = chat['agents']
            response_html = cls.render_response(responses[dialogue_id], agents)
            html_lines.extend(response_html)

        return completed, html_lines


    @classmethod
    def aggregate_chats(cls, transcripts, responses=None, css_file=None):
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
            completed, chat_html = cls.visualize_chat(chat, responses=responses, id_=idx)
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

    @classmethod
    def visualize_transcripts(cls, html_output, transcripts, responses=None, css_file=None):
        if not os.path.exists(os.path.dirname(html_output)) and len(os.path.dirname(html_output)) > 0:
            os.makedirs(os.path.dirname(html_output))

        html_lines = cls.aggregate_chats(transcripts, responses, css_file)

        outfile = open(html_output, 'w')
        for line in html_lines:
            outfile.write(line+"\n")
        outfile.close()

    @classmethod
    def write_chat_htmls(cls, transcripts, outdir, responses=None):
        outdir = os.path.join(outdir, 'chat_htmls')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        for chat in transcripts:
            dialogue_id = chat['uuid']
            _, chat_html = cls.visualize_chat(chat, responses=responses)
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

    @classmethod
    def write_metadata(cls, transcripts, outdir, responses=None):
        metadata = {'data': []}
        for chat in transcripts:
            if len(chat['events']) == 0:
                continue
            row = {}
            row['dialogue_id'] = chat['uuid']
            row['scenario_id'] = chat['scenario_uuid']
            scenario = cls.get_scenario(chat)
            row['num_items'] = len(scenario.kbs[0].items)
            row['num_attrs'] = len(scenario.attributes)
            row['outcome'] = 'fail' if chat['outcome']['reward'] == 0 else 'success'
            row['agent0'] = cls.agent_labels[chat['agents']['0']]
            row['agent1'] = cls.agent_labels[chat['agents']['1']]
            if responses:
                dialogue_response = responses[chat['uuid']]
                question_scores = defaultdict(list)
                for agent_id, scores in dialogue_response.iteritems():
                    for question in cls.questions:
                        question_scores[question].extend(scores[question])
                for question, scores in question_scores.iteritems():
                    row[question] = np.mean(scores)
            metadata['data'].append(row)
        write_json(metadata, os.path.join(outdir, 'metadata.json'))

    @classmethod
    def write_viewer_data(cls, html_output, transcripts, responses=None):
        if not os.path.exists(html_output):
            os.makedirs(html_output)
        cls.write_metadata(transcripts, html_output, responses)
        cls.write_chat_htmls(transcripts, html_output, responses)

    @classmethod
    def visualize(cls, viewer_mode, html_output, chats, responses=None, css_file=None):
        if viewer_mode:
            # External js and css
            cls.write_viewer_data(html_output, chats, responses=responses)
        else:
            # Inline style
            cls.visualize_transcripts(html_output, chats, css_file=css_file, responses=responses)

class MutualFriendsHTMLVisualizer(BaseHTMLVisualizer):
    agent_labels = {'human': 'Human', 'rulebased': 'Rule-based', 'static-neural': 'StanoNet', 'dynamic-neural': 'DynoNet'}
    questions = ("fluent", "correct", 'cooperative', "humanlike")

    @classmethod
    def render_scenario(cls, scenario):
        html = ["<div class=\"scenario\">", '<div class=\"divTitle\">Scenario %s</div>' % scenario.uuid]
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


class NegotiationHTMLVisualizer(BaseHTMLVisualizer):
    agent_labels = {'human': 'Human'}
    questions = ('fluent', 'honest', 'persuasive', 'fair')

    @classmethod
    def render_scenario(cls, scenario):
        html = ["<div class=\"scenario\">", '<div class=\"divTitle\">Scenario %s</div>' % scenario.uuid]
        for (idx, kb) in enumerate(scenario.kbs):
            kb_dict = kb.to_dict()
            html.append("<div class=\"kb%d\"><table><tr>"
                        "<td colspan=\"2\" class=\"agentLabel\">Agent %d</td></tr>" % (idx, idx))

            html.append("<tr><th colspan=\"2\">Personal Attributes</th></tr>")

            for attr in kb_dict['personal'].keys():
                html.append("<tr><td>%s</td><td>%s</td></tr>" % (attr, kb_dict['personal'][attr]))

            html.append("<tr><th colspan=\"2\">Object Attributes</th></tr>")
            for attr in kb_dict['item'].keys():
                entity = kb_dict['item'][attr]
                if entity is None:
                    entity = "?"
                elif isinstance(entity, list):
                    entity = ", ".join([str(x) for x in entity])
                html.append("<tr><td>%s</td><td>%s</td></tr>" % (attr, entity))
            html.append("</table></div>")

        html.append("</div>")
        return html
