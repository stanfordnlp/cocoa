import os
import numpy as np
import json
import datetime

from cocoa.analysis.html_visualizer import HTMLVisualizer as BaseHTMLVisualizer
from cocoa.analysis.utils import reject_transcript
from cocoa.core.event import Event

class HTMLVisualizer(BaseHTMLVisualizer):
    agent_labels = {'human': 'Human', 'rulebased': 'Rule-based', 'config-rulebased': 'Config-rulebased', 'neural': 'Neural'}
    #questions = ('fluent', 'negotiator', 'persuasive', 'fair', 'coherent')
    questions = ('negotiator',)

    @classmethod
    def render_scenario(cls, scenario, img_path=None, kbs=None, uuid=None):
        # Sometimes we want to directly give it elements of the scenario
        uuid = scenario.uuid if uuid is None else uuid
        kbs = kbs or scenario.kbs
        # kbs[0] = rulebaed agent, kbs[1] = human
        html = ["<div class=\"scenario\">", '<div class=\"divTitle\">Scenario %s</div>' % uuid]
        # Post (display the seller's (full) KB)
        items = ["book", "hat", "ball"]
        column_headers = ["Name", "Count", "Value"]
        item_counts = kbs[0].item_counts

        # Private info
        for idx, kb in enumerate(kbs):
            html.append("<div class=\"kb%d\"><table><tr>"
                        "<td colspan=\"3\" class=\"agentLabel\">Agent %d</td></tr>" % (idx, idx))
            html.append(('<tr>%s</tr>' % (''.join(['<th>%s</th>' % x for x in column_headers]))))
            for item in items:
                row_values = [item.title(), kb.item_counts[item], kb.item_values[item]]
                html.append(('<tr>%s</tr>' % (''.join(['<td>%s</td>' % x for x in row_values]))))
            html.append("</table></div>")

        html.append("</div>")
        return html

    @classmethod
    def visualize_transcripts(cls, html_output, transcripts, responses=None, css_file=None, img_path=None):
        if not os.path.exists(os.path.dirname(html_output)) and len(os.path.dirname(html_output)) > 0:
            os.makedirs(os.path.dirname(html_output))

        html_lines = cls.aggregate_chats(transcripts, responses, css_file, img_path)

        outfile = open(html_output, 'w')
        for line in html_lines:
            outfile.write(line.encode('utf8')+"\n")
        outfile.close()

    @classmethod
    def render_chat(cls, chat, agent=None, partner_type='human', workers_ids=None):
        events = Event.gather_eval([Event.from_dict(e) for e in chat["events"]])

        if len(events) == 0:
            return False, False, None

        chat_html= ['<div class=\"chatLog\">',
                '<div class=\"divTitle\"> Chat Log: %s </div>' % (chat['uuid']),
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
            if event.action == 'message':
                s = event.data
            elif event.action == 'select':
                s = 'DEAL AGREED: {}'.format(event.data)
            elif event.action == 'reject':
                s = 'NO DEAL'
            elif event.action == 'eval':
                s = 'EVAL {utterance} || {tags}'.format(utterance=event.data['utterance'], tags=' '.join([k for k, v in event.data['labels'].iteritems() if v == 1]))
            else:
                continue

            try:
                tags = ', '.join(event.tags)
            except AttributeError:
                tags = ''

            if event.template is None:
                response_tag = ''
                count = ''
                template = ''
            else:
                response_tag = event.template['response_tag'] + '|' + event.template['source']
                count = np.log(event.template['count'])
                template = event.template['response']

            row = '<tr class=\"agent%d\">\
                    <td class=\"time\">%s</td>\
                    <td class=\"agent\">%s</td>\
                    <td class=\"tags\">%s</td>\
                    <td class=\"act\">%s</td>\
                    <td class=\"count\">%s</td>\
                    <td class=\"template\">%s</td>\
                    <td class=\"message\">%s</td>\
                   </tr>' % (event.agent, t, a, tags, response_tag, count, template, s)
            chat_html.append(row)

        chat_html.extend(['</table>', '</div>'])
        completed = False if chat["outcome"] is None or chat["outcome"]["reward"] == 0 else True

        # Show config and results
        if chat.get('agents_info') is not None:
            chat_html.append('<p>Bot config: {}</p>'.format(str(chat['agents_info']['config'])))
        rejected = reject_transcript(chat, 0) and reject_transcript(chat, 1)

        return completed, rejected, chat_html

    @classmethod
    def _render_response(cls, response, agent_id, agent):
        html = []
        html.append('<table class=\"response%d\">' % agent_id)
        html.append('<tr><td colspan=\"3\" class=\"agentLabel\">Response to agent %d (%s)</td></tr>' % (agent_id, cls.agent_labels[agent]))
        html.append('<tr>%s</tr>' % (''.join(['<th>%s</th>' % x for x in ('Question', 'Mean', 'Response')])))
        for question in cls.questions:
            if question not in response:
                continue
            scores = response[question]
            raw_score = ' / '.join([str(x) for x in scores])
            question_html = ('<td>%s</td>' % question)
            mean_html = ('<td style=\"text-align:center\">%s</td>' % np.mean(scores))
            raw_score_html = ('<td style=\"text-align:center\">%s</td>' % np.mean(scores))
            column_html = [ x for x in (question_html, mean_html, raw_score_html)]
            html.append('<tr>%s</tr>' % (''.join(column_html)))

        if 'comments' in response:
            comment_str = response['comments'][0]
            if len(comment_str) > 0:
                html.append('<tr><td>%s</td><td colspan=2>%s</td></tr>' % ('comments', comment_str))

        html.append('</table>')
        return html

    @classmethod
    def visualize(cls, viewer_mode, html_output, chats, responses=None, css_file=None, img_path=None):
        if viewer_mode:
            # External js and css
            cls.write_viewer_data(html_output, chats, responses=responses)
        else:
            # Inline style
            cls.visualize_transcripts(html_output, chats,
                    css_file=css_file, responses=responses, img_path=img_path)
