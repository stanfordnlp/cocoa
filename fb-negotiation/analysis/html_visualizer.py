import os

from cocoa.analysis.html_visualizer import HTMLVisualizer as BaseHTMLVisualizer
from cocoa.analysis.utils import reject_transcript

class HTMLVisualizer(BaseHTMLVisualizer):
    agent_labels = {'human': 'Human', 'rulebased': 'Rule-based', 'config-rulebased': 'Config-rulebased'}
    #questions = ('fluent', 'negotiator', 'persuasive', 'fair', 'coherent')
    questions = ('negotiator',)

    @classmethod
    def render_scenario(cls, scenario, img_path=None, kbs=None, uuid=None):
        # Sometimes we want to directly give it elements of the scenario
        uuid = scenario.uuid if uuid is None else uuid
        kbs = kbs or scenario.kbs

        html = ["<div class=\"scenario\">", '<div class=\"divTitle\">Scenario %s</div>' % uuid]
        # Post (display the seller's (full) KB)
        if kbs[0].facts['personal']['Role'] == 'seller':
            facts = kbs[0].facts
        else:
            facts = kbs[1].facts
        html.append("<p><b>%s ($%d)</b></p>" % (facts['item']['Title'], facts['item']['Price']))
        html.append("<p>%s</p>" % '<br>'.join(facts['item']['Description']))
        if img_path and len(facts['item']['Images']) > 0:
            html.append("<p><img src=%s></p>" % os.path.join(img_path, facts['item']['Images'][0]))
        # Private info
        for (idx, kb) in enumerate(kbs):
            kb_dict = kb.to_dict()
            html.append("<div class=\"kb%d\"><table><tr>"
                        "<td colspan=\"2\" class=\"agentLabel\">Agent %d</td></tr>" % (idx, idx))

            html.append("<tr><th colspan=\"2\">Personal Attributes</th></tr>")

            for attr in kb_dict['personal'].keys():
                html.append("<tr><td>%s</td><td>%s</td></tr>" % (attr, kb_dict['personal'][attr]))

            html.append("</table></div>")

        html.append("</div>")
        return html

    @classmethod
    def render_chat(cls, chat, agent=None, partner_type='human'):
        complete, _, html_lines = super(HTMLVisualizer, cls).render_chat(chat, agent=agent, partner_type=partner_type)
        # Show config and results
        if chat.get('agents_info') is not None:
            html_lines.append('<p>Bot config: {}</p>'.format(str(chat['agents_info']['config'])))
        rejected = reject_transcript(chat, 0) and reject_transcript(chat, 1)
        return complete, rejected, html_lines

    @classmethod
    def visualize(cls, viewer_mode, html_output, chats, responses=None, css_file=None, img_path=None):
        if viewer_mode:
            # External js and css
            cls.write_viewer_data(html_output, chats, responses=responses)
        else:
            # Inline style
            cls.visualize_transcripts(html_output, chats, css_file=css_file, responses=responses, img_path=img_path)
