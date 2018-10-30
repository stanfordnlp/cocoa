import os
import numpy as np
import json
import datetime
from itertools import izip
from collections import defaultdict
from argparse import ArgumentParser

from cocoa.core.scenario_db import ScenarioDB
from cocoa.core.schema import Schema
from cocoa.core.event import Event
from cocoa.core.util import write_json, read_json

from core.scenario import Scenario


class HTMLVisualizer(object):
    agent_labels = None
    questions = None

    @classmethod
    def add_html_visualizer_arguments(cls, parser):
        parser.add_argument('--html-output', help='Name of directory to write HTML report to')
        parser.add_argument('--viewer-mode', action='store_true', help='Output viewer instead of single html')
        parser.add_argument('--css-file', default='../chat_viewer/css/my.css', help='css for tables/scenarios and chat logs')
        parser.add_argument('--img-path', help='path to images')

    @classmethod
    def get_scenario(cls, chat):
        scenario = Scenario.from_dict(None, chat['scenario'])
        return scenario

    @classmethod
    def render_event(cls, event):
        s = None
        if event.action == 'message':
            s = event.data
        elif event.action == 'eval':
            s = 'EVAL {utterance} || {tags}'.format(utterance=event.data['utterance'], tags=' '.join([k for k, v in event.data['labels'].iteritems() if v == 1]))
        return s

    @classmethod
    def render_chat(cls, chat, agent=None, partner_type='human', worker_ids=None):
        events = Event.gather_eval([Event.from_dict(e) for e in chat["events"]])

        if len(events) == 0:
            return False, False, None

        def get_worker_id(agent_id):
            if worker_ids is None:
                return 'N/A'
            id_ = worker_ids.get(str(agent_id), None)
            return id_ if id_ is not None else ''

        chat_html= ['<div class=\"chatLog\">',
                '<div class=\"divTitle\"> Chat Log: %s <br> Agent 0: %s Agent 1: %s </div>' % (chat['uuid'], get_worker_id(0), get_worker_id(1)),
                '<table class=\"chat\">']

        # Used for visualizing chat during debugging
        agent_str = {0: '', 1: ''}
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
            if not event.time:
                t = None
            else:
                t = datetime.datetime.fromtimestamp(float(event.time)).strftime('%Y-%m-%d %H:%M:%S')
            a = agent_str[event.agent]
            s = cls.render_event(event)
            if s is None:
                continue

            try:
                tags = ', '.join(event.tags)
            except AttributeError:
                tags = ''

            if not isinstance(event.metadata, dict):
                response_tag = ''
                template = ''
                received_row = None
            else:
                sent_data = event.metadata['sent']
                response_tag = sent_data['logical_form']['intent']
                template = sent_data['template']
                if isinstance(template, dict):
                    template = template['template']

                # Received event
                received_data = event.metadata['received']
                partner_tag = received_data['logical_form']['intent']
                partner_template = ' '.join(received_data['template'])
                received_row = '<tr class=\"agent%d\">\
                        <td class=\"time\">%s</td>\
                        <td class=\"agent\">%s</td>\
                        <td class=\"tags\">%s</td>\
                        <td class=\"act\">%s</td>\
                        <td class=\"template\">%s</td>\
                        <td class=\"message\">%s</td>\
                       </tr>' % (event.agent, '', '', '', partner_tag, partner_template, '')

            row = '<tr class=\"agent%d\">\
                    <td class=\"time\">%s</td>\
                    <td class=\"agent\">%s</td>\
                    <td class=\"tags\">%s</td>\
                    <td class=\"act\">%s</td>\
                    <td class=\"template\">%s</td>\
                    <td class=\"message\">%s</td>\
                   </tr>' % (event.agent, t, a, tags, response_tag, template, s)
            if received_row:
                chat_html.append(received_row)
            chat_html.append(row)

        chat_html.extend(['</table>', '</div>'])

        completed = False if chat["outcome"] is None or chat["outcome"]["reward"] == 0 else True

        return completed, True, chat_html

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
    def visualize_chat(cls, chat, agent=None, partner_type='Human', responses=None, id_=None, img_path=None, worker_ids=None):
        chat_worker_ids = worker_ids.get(chat['uuid'], None) if worker_ids else None
        completed, rejected, chat_html = cls.render_chat(chat, agent, partner_type, chat_worker_ids)
        if chat_html is None:
            return False, False, None

        html_lines = []

        scenario_html = cls.render_scenario(cls.get_scenario(chat), img_path)
        html_lines.extend(scenario_html)

        html_lines.extend(chat_html)

        dialogue_id = chat['uuid']
        if responses and dialogue_id in responses:
            agents = chat['agents']
            response_html = cls.render_response(responses[dialogue_id], agents)
            html_lines.extend(response_html)

        return completed, rejected, html_lines


    @classmethod
    def aggregate_chats(cls, transcripts, responses=None, css_file=None, img_path=None, worker_ids=None):
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

        accepted_chats = {'completed': [], 'incompleted': []}
        rejected_chats = {'completed': [], 'incompleted': []}
        num_accepted = {'completed': 0, 'incompleted': 0}
        num_rejected = {'completed': 0, 'incompleted': 0}
        total = 0
        def add_chat(chats, chat_html):
            chats.extend(chat_html)
            chats.append('</div>')
            chats.append("<hr>")
        transcripts = sorted(transcripts, key=lambda x: x['events'][0]['time'], reverse=True)
        # transcripts = sorted(transcripts, key=lambda x: x['scenario']['post_id'], reverse=True)
        for (idx, chat) in enumerate(transcripts):
            completed, rejected, chat_html = cls.visualize_chat(chat, responses=responses, id_=idx, img_path=img_path, worker_ids=worker_ids)
            if chat_html is None:
                continue
            k = 'completed' if completed else 'incompleted'
            if not rejected:
                add_chat(accepted_chats[k], chat_html)
                num_accepted[k] += 1
            else:
                add_chat(rejected_chats[k], chat_html)
                num_rejected[k] += 1
            total += 1

        naccepted = sum(num_accepted.values())
        nrejected = sum(num_rejected.values())
        html.extend([
            '<h3>Total number of chats: %d</h3>' % total,
            '<h3>Number of chats accepted: %d (completed %d, incompleted %d)</h3>' % (naccepted, num_accepted['completed'], num_accepted['incompleted']),
            '<h3>Number of chats rejected: %d (completed %d, incompleted %d)</h3>' % (nrejected, num_rejected['completed'], num_rejected['incompleted'])])
        html.append('<hr>')
        html.extend(accepted_chats['completed'])
        html.extend(accepted_chats['incompleted'])
        html.extend(rejected_chats['completed'])
        html.extend(rejected_chats['incompleted'])
        html.append('</body></html>')
        return html

    @classmethod
    def visualize_transcripts(cls, html_output, transcripts, responses=None, css_file=None, img_path=None, worker_ids=None):
        if not os.path.exists(os.path.dirname(html_output)) and len(os.path.dirname(html_output)) > 0:
            os.makedirs(os.path.dirname(html_output))

        html_lines = cls.aggregate_chats(transcripts, responses, css_file, img_path, worker_ids)

        outfile = open(html_output, 'w')
        for line in html_lines:
            outfile.write(line.encode('utf8')+"\n")
        outfile.close()

    @classmethod
    def write_chat_htmls(cls, transcripts, outdir, responses=None):
        outdir = os.path.join(outdir, 'chat_htmls')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        for chat in transcripts:
            dialogue_id = chat['uuid']
            _, _, chat_html = cls.visualize_chat(chat, responses=responses)
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
    def visualize(cls, viewer_mode, html_output, chats, responses=None, css_file=None, img_path=None, worker_ids=None):
        if viewer_mode:
            # External js and css
            cls.write_viewer_data(html_output, chats, responses=responses)
        else:
            # Inline style
            cls.visualize_transcripts(html_output, chats, css_file=css_file, responses=responses, img_path=img_path, worker_ids=worker_ids)
