from cocoa.model.dialogue_state import DialogueState as State

class DialogueState(State):
    def __init__(self, agent, kb):
        super(DialogueState, self).__init__(agent, kb)
        self.curr_title = None
        self.mentioned_titles = set()

    def update(self, agent, utterance):
        super(DialogueState, self).update(agent, utterance)
        if not utterance:
            return
        # Start new movie
        #if agent == self.partner and utterance.lf.intent == 'ask-you':
        #    self.curr_title = None
        #    print 'update curr_title', self.curr_title
        lf = utterance.lf
        if hasattr(lf, 'entities') and lf.entities is not None:
            for entity in lf.entities:
                if entity.canonical.type == 'title':
                    self.curr_title = entity.canonical.value
                    self.mentioned_titles.add(self.curr_title)
            #print 'update curr_title', self.curr_title

