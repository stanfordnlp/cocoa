from parser import LogicalForm as LF, Utterance

class DialogueState(object):
    def __init__(self, agent, kb):
        self.agent = agent
        self.partner = 1 - agent
        self.kb = kb
        self.time = 0
        init_utterance = Utterance(logical_form=LF('<start>'), template=['<start>'])
        self.utterance = [init_utterance, init_utterance]
        self.done = set()

    @property
    def my_act(self):
        return self.utterance[self.agent].lf.intent

    @property
    def partner_act(self):
        return self.utterance[self.partner].lf.intent

    @property
    def partner_utterance(self):
        return self.utterance[self.partner]

    @property
    def partner_template(self):
        try:
            return self.utterance[self.partner].template
        except:
            return None

    def update(self, agent, utterance):
        if not utterance:
            return
        self.time += 1
        self.utterance[agent] = utterance
        if agent == self.agent:
            self.done.add(utterance.lf.intent)

