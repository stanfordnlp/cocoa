import numpy as np
from collections import defaultdict

from cocoa.core.dataset import Example
from cocoa.analysis.visualizer import Visualizer as BaseVisualizer

from core.scenario import Scenario
from analysis.html_visualizer import HTMLVisualizer

class Visualizer(BaseVisualizer):
    agents = ('human', 'rulebased', 'config-rule')
    agent_labels = {'human': 'Human', 'rulebased': 'Rule-based',
            'sl-words': 'SL-words',
            'rl-words-margin': 'RL-words-margin',
            'rl-words-length': 'RL-words-length',
            'rl-words-fair': 'RL-words-fair',
            'sl-states': 'SL-states',
            'rl-states-margin': 'RL-states-margin',
            'rl-states-length': 'RL-states-length',
            'rl-states-fair': 'RL-states-fair',
            }
    #questions = ('fluent', 'negotiator', 'persuasive', 'fair', 'coherent')
    questions = ('negotiator',)
    question_labels = {"negotiator": 'Humanlikeness'}

    def __init__(self, chats, surveys=None):
        super(Visualizer, self).__init__(chats, surveys)
        mask = None
        self.question_scores = None
        if surveys:
            self.agents, self.question_scores = self.read_eval(self.surveys, mask)

    def question_type(self, question):
        if question == 'comments':
            return 'str'
        else:
            return 'num'

    def filter(self, deprecated):
        print("tried to filter for #bad_worker_ids, this has been deprecated")
        return deprecated

    def html_visualize(self, viewer_mode, html_output, css_file=None, img_path=None, worker_ids=None):
        chats = []
        scenario_to_chats = defaultdict(set)
        dialogue_responses = None
        if self.question_scores:
            dialogue_responses = self.get_dialogue_responses(self.question_scores)

            # Put chats in the order of responses
            chats_with_survey = set()
            for dialogue_id, agent_responses in dialogue_responses.iteritems():
                chat = self.uuid_to_chat[dialogue_id]
                scenario_id = chat['scenario_uuid']
                chats.append((scenario_id, chat))
                chats_with_survey.add(dialogue_id)
                scenario_to_chats[scenario_id].add(dialogue_id)
            chats = [x[1] for x in sorted(chats, key=lambda x: x[0])]
            # Incomplete chats (redirected, no survey)
            for (dialogue_id, chat) in self.uuid_to_chat.iteritems():
                if dialogue_id not in chats_with_survey:
                    chats.append(chat)
        else:
            for (dialogue_id, chat) in self.uuid_to_chat.iteritems():
                scenario_id = chat['scenario_uuid']
                chats.append((scenario_id, chat))
                scenario_to_chats[scenario_id].add(dialogue_id)
            chats = [x[1] for x in sorted(chats, key=lambda x: x[0])]

        html_visualizer = HTMLVisualizer()
        html_visualizer.visualize(viewer_mode, html_output, chats,
            responses=dialogue_responses, css_file=css_file, img_path=img_path)

    def print_results(self, results):
        systems = sorted(results.keys())
        print '{:<20s} {:<10s} {:<10s} {:<10s} {:<10s} {:<10s}'.format('system', '%agree', 'a_points', 't_points', 'length', '#examples')
        def points_to_str(points):
            return '{me:<.1f}-{partner:<.1f}'.format(me=points['agent'], partner=points['partner'])
        for system in systems:
            res = results[system]
            print '{:<20s} {:<10.2f} {:<10s} {:<10s} {:<10.2f} {:<10d}'.format(
                    system,
                    res['% agreed'],
                    points_to_str(res['agreed points']),
                    points_to_str(res['total points']),
                    res['average length'],
                    res['num examples'],
                    )

    def compute_effectiveness_for_system(self, examples, system):
        num_agreed = 0
        total = 0
        agreed_points = {'agent': 0, 'partner': 0}
        total_points = {'agent': 0, 'partner': 0}
        num_mismatch = 0
        num_nodeal = 0
        length = 0
        for ex in examples:
            if system == 'human':
                # Take human winner
                #rewards = [ex.outcome['reward'][str(agent)] for agent in (0, 1)]
                #eval_agents = [np.argmax(rewards)]
                # Both
                #eval_agents = [0, 1]
                # Random sample
                eval_agents = [np.random.randint(2)]
            else:
                eval_agents = [0 if ex.agents[0] == system else 1]

            for eval_agent in eval_agents:
                l = len([e for e in ex.events if e.action == 'message'])
                length += l
                total += 1
                if ex.outcome.get('valid_deal'):
                    num_agreed += 1
                    point = ex.outcome['reward'][str(eval_agent)]
                    partner_point = ex.outcome['reward'][str(1-eval_agent)]
                    agreed_points['agent'] += point
                    agreed_points['partner'] += partner_point
                    total_points['agent'] += point
                    total_points['partner'] += partner_point
                else:
                    if ex.outcome.get('agreed'):
                        num_mismatch += 1
                    else:
                        num_nodeal += 1

        for k in agreed_points:
            try:
                agreed_points[k] /= float(num_agreed)
            except ZeroDivisionError:
                agreed_points[k] = 0
        for k in total_points:
            try:
                total_points[k] /= float(total)
            except ZeroDivisionError:
                total_points[k] = 0

        result = {'% agreed': num_agreed / float(total) if total > 0 else 0,
                'agreed points': agreed_points,
                'total points': total_points,
                'mismatch': num_mismatch,
                'no deal': num_nodeal,
                'average length': length / (float(len(examples)) + 1e-5),
                'num examples': len(examples),
                }

        #print system.upper()
        #for k, v in result.iteritems():
        #    if k == 'agreed points' or k == 'total points':
        #        points = 'partner={:.2f} agent={:.2f}'.format(v['partner'], v['agent'])
        #        print '{:<15s} {}'.format(k, points)
        #    else:
        #        print '{:<15s} {:.2f}'.format(k, v)

        return result

