from parser import LogicalForm as LF

class DialogueState(object):
    def __init__(self, agent, kb):
        self.agent = agent
        self.partner = 1 - agent
        self.kb = kb
        self.time = 0
        self.act = [LF('<start>'), LF('<start>')]
        self.done = set()

    @property
    def my_act(self):
        return self.act[self.agent].intent if self.act[self.agent] else None

    @property
    def partner_act(self):
        return self.act[self.partner].intent if self.act[self.partner] else None

    def update(self, agent, utterance):
        self.time += 1
        self.act[agent] = utterance.lf
        self.done.add(utterance.lf.intent)
