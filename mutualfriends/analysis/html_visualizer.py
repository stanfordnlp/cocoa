from cocoa.analysis.html_visualizer import HTMLVisualizer as BaseHTMLVisualizer

class HTMLVisualizer(BaseHTMLVisualizer):
    agent_labels = {'human': 'Human', 'rulebased': 'Rule-based', 'static-neural': 'StanoNet', 'dynamic-neural': 'DynoNet', 'rule_bot': 'Rule-based', 'neural': 'Neural'}
    #questions = ("fluent", "correct", 'cooperative', "humanlike")
    questions = ('cooperative', "humanlike")

    @classmethod
    def render_scenario(cls, scenario, img_path=None):
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

    @classmethod
    def render_event(cls, event):
        if event.action == 'select':
            s = 'SELECT (' + ' || '.join(event.data.values()) + ')'
        else:
            s = super(HTMLVisualizer, cls).render_event(event)
        return s
