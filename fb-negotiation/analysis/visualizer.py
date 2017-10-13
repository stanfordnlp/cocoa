from collections import defaultdict

from cocoa.core.dataset import Example
from cocoa.analysis.visualizer import Visualizer as BaseVisualizer

from core.scenario import Scenario
from analysis.html_visualizer import HTMLVisualizer

class Visualizer(BaseVisualizer):
    agents = ('human', 'rulebased', 'config-rule')
    agent_labels = {'human': 'Human', 'rulebased': 'Rule-based', 'config-rulebased': 'Config-rulebased'}
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

    def compute_effectiveness(self):
        chats = defaultdict(list)
        for raw in self.chats:
            ex = Example.from_dict(raw, Scenario)
            if ex.agents[0] == 'human' and ex.agents[1] == 'human':
                chats['human'].append(ex)
            elif ex.agents[0] != 'human':
                chats[ex.agents[0]].append(ex)
            elif ex.agents[1] != 'human':
                chats[ex.agents[1]].append(ex)

        results = {}
        for system, examples in chats.iteritems():
            if system == 'human':
                continue
            results[system] = self._compute_effectiveness(examples, system)
            print system, results[system]

    def is_complete(self, ex):
        if ex.outcome is not None and ex.outcome.get('valid_deal') is not None:
            return True
        return False

    def _compute_effectiveness(self, examples, system):
        num_agreed = 0
        total = 0
        agreed_points = 0
        total_points = 0
        num_mismatch = 0
        num_nodeal = 0
        for ex in examples:
            if not self.is_complete(ex):
                continue
            if ex.agents[0] == system:
                eval_agent = 0
            else:
                eval_agent = 1
            total += 1
            if ex.outcome.get('valid_deal'):
                num_agreed += 1
                point = ex.outcome['item_split'][str(eval_agent)]['reward']
                agreed_points += point
                total_points += point
            else:
                if ex.outcome['item_split']['1'] is None or ex.outcome['item_split']['0'] is None:
                    num_nodeal += 1
                else:
                    num_mismatch += 1

        return {'% agreed': num_agreed / float(total),
                'agreed points': agreed_points / float(num_agreed),
                'total points': total_points / float(total),
                'total': total,
                'mismatch': num_mismatch,
                'no deal': num_nodeal,
                }

