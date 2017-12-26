from collections import defaultdict
from cocoa.analysis.utils import safe_div
from cocoa.analysis.visualizer import Visualizer as BaseVisualizer

class Visualizer(BaseVisualizer):
    agents = ('human', 'rulebased', 'static-neural', 'dynamic-neural', 'neural')
    agent_labels = {'human': 'Human', 'rulebased': 'Rule-based', 'static-neural': 'StanoNet', 'dynamic-neural': 'DynoNet', 'neural': 'Neural'}
    questions = ("fluent", "correct", 'cooperative', "humanlike")
    question_labels = {"fluent": 'Fluency', "correct": 'Correctness', 'cooperative': 'Cooperation', "humanlike": 'Human-likeness'}
    colors = ("#33a02c", "#b2df8a", "#1f78b4", "#a6cee3")

    def __init__(self, chats, surveys=None, worker_ids=None):
        super(Visualizer, self).__init__(chats, surveys, worker_ids)
        if surveys:
            mask = self.filter(self.surveys)
            self.agents, self.question_scores = self.read_eval(self.surveys, mask)

    def question_type(self, question):
        if question == 'comments' or question.endswith('text'):
            return 'str'
        else:
            return 'num'

    def compute_effectiveness_for_system(self, examples, system):
        num_success = 0
        total = 0
        chat_ids_with_survey = self.surveys[0][1].keys()
        n_select = 0
        n_turns = 0
        for ex in examples:
            # TODO
            if ex.ex_id in chat_ids_with_survey:
                n_select += sum([1 if e.action == 'select' else 0 for e in ex.events])
                n_turns += len(ex.events)
                total += 1
                if ex.outcome['reward'] == 1:
                    num_success += 1

        result = {
                'success per select': safe_div(num_success, float(n_select)),
                'success per turn': safe_div(num_success, float(n_turns)),
                'success': safe_div(num_success, float(total)),
                }

        print system.upper()
        for k, v in result.iteritems():
            print '{:<15s} {:.2f}'.format(k, v)

        return result


    def filter(self, raw_evals):
        '''
        Only keep scenarios where all 4 agents are evaluated.
        '''
        # TODO
        return None
        scenario_to_agents = defaultdict(set)
        scenario_to_chats = defaultdict(set)
        for eval_ in raw_evals:
            dialogue_agents = eval_[0]
            dialogue_scores = eval_[1]
            for dialogue_id, agent_dict in dialogue_agents.iteritems():
                chat = self.uuid_to_chat[dialogue_id]
                scenario_id = chat['scenario_uuid']
                if isinstance(agent_dict, basestring):
                    agent_dict = eval(agent_dict)
                scores = dialogue_scores[dialogue_id]
                for agent_id, results in scores.iteritems():
                    if len(results) == 0:
                        continue
                    agent_type = agent_dict[str(agent_id)]
                    scenario_to_agents[scenario_id].add(agent_type)
                    scenario_to_chats[scenario_id].add(dialogue_id)
        good_dialogues = []  # with 4 agent types
        print 'Total scenarios:', len(scenario_to_agents)
        print 'Good scenarios:', sum([1 if len(a) == 4 else 0 for s, a in scenario_to_agents.iteritems()])
        for scenario_id, agents in scenario_to_agents.iteritems():
            if len(agents) == 4:
                good_dialogues.extend(scenario_to_chats[scenario_id])
                #assert len(scenario_to_chats[scenario_id]) >= 4
        filtered_dialogues = set(good_dialogues)
        return filtered_dialogues
