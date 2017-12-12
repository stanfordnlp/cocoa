from collections import defaultdict

from cocoa.core.dataset import Example
from cocoa.analysis.visualizer import Visualizer as BaseVisualizer

from core.scenario import Scenario
from analyze_strategy import StrategyAnalyzer

class Visualizer(BaseVisualizer):
    agents = ('human', 'rulebased', 'config-rule', 'neural-gen')
    agent_labels = {'human': 'Human', 'rulebased': 'Rule-based', 'config-rulebased': 'Config-rulebased', 'neural-gen': 'Neural'}
    #questions = ('fluent', 'negotiator', 'persuasive', 'fair', 'coherent')
    questions = ('negotiator',)
    question_labels = {"fluent": 'Fluency', "negotiator": 'Humanlikeness', 'persuasive': 'Persuasiveness', "fair": 'Fairness', 'coherent': 'Coherence'}

    def __init__(self, chats, surveys=None, worker_ids=None):
        super(Visualizer, self).__init__(chats, surveys, worker_ids)
        mask = None
        self.question_scores = None
        if surveys:
            self.agents, self.question_scores = self.read_eval(self.surveys, mask)

    def question_type(self, question):
        if question == 'comments':
            return 'str'
        else:
            return 'num'

    def compute_effectiveness_for_system(self, examples, system):
        num_success = 0
        final_offer = 0
        for ex in examples:
            if not StrategyAnalyzer.has_deal(ex):
                continue
            if ex.agents[0] == system:
                eval_agent = 0
            else:
                eval_agent = 1
            role = ex.scenario.kbs[eval_agent].facts['personal']['Role']
            num_success += 1
            final_price = ex.outcome['offer']['price']
            margin = StrategyAnalyzer.get_margin(ex, final_price, eval_agent, role)
            if not margin:
                continue
            else:
                final_offer += margin
        return {'success rate': num_success / float(len(examples)),
                'average margin': final_offer / float(num_success),
                }
