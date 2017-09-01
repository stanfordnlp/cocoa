from cocoa.analysis.visualizer import Visualizer as BaseVisualizer

class Visualizer(BaseVisualizer):
    agents = ('human', 'rulebased', 'static-neural', 'dynamic-neural')
    agent_labels = {'human': 'Human', 'rulebased': 'Rule-based', 'static-neural': 'StanoNet', 'dynamic-neural': 'DynoNet'}
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

    def filter(self, raw_evals):
        '''
        Only keep scenarios where all 4 agents are evaluated.
        '''
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
