import os
import numpy as np
import json
import datetime

from cocoa.analysis.html_visualizer import HTMLVisualizer as BaseHTMLVisualizer
from cocoa.analysis.utils import reject_transcript
from cocoa.core.event import Event

class HTMLVisualizer(BaseHTMLVisualizer):
    agent_labels = {'human': 'Human', 'rulebased': 'Rule-based', 'config-rulebased': 'Config-rulebased'}
    questions = ('humanlike', 'interesting')

    @classmethod
    def render_scenario(cls, scenario, img_path=None, kbs=None, uuid=None):
        # Sometimes we want to directly give it elements of the scenario
        uuid = scenario.uuid if uuid is None else uuid
        kbs = kbs or scenario.kbs
        # kbs[0] = rulebaed agent, kbs[1] = human
        html = ["<div class=\"scenario\">", '<div class=\"divTitle\">Scenario %s %s</div>' % (uuid, kbs[0].topic['Topic'])]
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
    def render_event(cls, event):
        if event.action == 'done':
            s = 'DONE TALKING'
        else:
            s = super(HTMLVisualizer, cls).render_event(event)
        return s

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
