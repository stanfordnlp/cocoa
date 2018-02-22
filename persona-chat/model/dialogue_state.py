from cocoa.model.dialogue_state import DialogueState as State

class DialogueState(State):
    def __init__(self, agent, kb):
        super(DialogueState, self).__init__(agent, kb)
        # no actual code for detecting at the moment
        self.current_topic = None

    def update(self, agent, utterance):
        super(DialogueState, self).update(agent, utterance)
        lf = utterance.lf
        if hasattr(lf, 'topic') and lf.topic is not None:
            self.current_topic = lf.topic
