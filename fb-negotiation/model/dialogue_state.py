from cocoa.model.dialogue_state import DialogueState as State

class DialogueState(State):
    def __init__(self, agent, kb):
        super(DialogueState, self).__init__(agent, kb)
        self.proposal = [None, None]
        self.curr_proposal = None

    @property
    def my_proposal(self):
        return self.proposal[self.agent]

    @my_proposal.setter
    def my_proposal(self, proposal):
        self.proposal[self.agent] = proposal

    @property
    def partner_proposal(self):
        return self.proposal[self.partner]

    def update(self, agent, utterance):
        super(DialogueState, self).update(agent, utterance)
        lf = utterance.lf
        if hasattr(lf, 'proposal') and lf.proposal is not None:
            self.proposal[agent] = lf.proposal
            self.curr_proposal = lf.proposal
