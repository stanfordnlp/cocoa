from collections import defaultdict
import numpy as np

from cocoa.core.dataset import Example
from cocoa.analysis.visualizer import Visualizer as BaseVisualizer

from core.scenario import Scenario
from analyze_strategy import StrategyAnalyzer

class Visualizer(BaseVisualizer):
    agents = ('human', 'rulebased', 'config-rule', 'neural-gen')
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

    def print_results(self, results):
        systems = sorted(results.keys())
        print '{:<20s} {:<10s} {:<10s} {:<10s} {:<10s}'.format('system', 'success', 'margin', 'length', '#examples')
        for system in systems:
            res = results[system]
            print '{:<20s} {:<10.2f} {:<10.2f} {:<10.2f} {:<10d}'.format(
                    system,
                    res['success rate'],
                    res['average margin'],
                    res['average length'],
                    res['num examples'],
                    )

    def compute_effectiveness_for_system(self, examples, system):
        num_success = 0
        final_offer = 0
        length = 0
        total = 0
        for ex in examples:
            if system == 'human':
                #eval_agents = [0, 1]
                eval_agents = [np.random.randint(2)]
            else:
                eval_agents = [0 if ex.agents[0] == system else 1]
            for eval_agent in eval_agents:
                total += 1
                l = len([e for e in ex.events if e.action == 'message'])
                length += l
                if not StrategyAnalyzer.has_deal(ex):
                    continue
                role = ex.scenario.kbs[eval_agent].facts['personal']['Role']
                num_success += 1
                final_price = ex.outcome['offer']['price']
                margin = StrategyAnalyzer.get_margin(ex, final_price, eval_agent, role)
                if not margin:
                    continue
                else:
                    final_offer += margin
        return {'success rate': num_success / (float(total) + 1e-5),
                'average margin': final_offer / (float(num_success) + 1e-5),
                'average length': length / (float(total) + 1e-5),
                'num examples': len(examples),
                }
