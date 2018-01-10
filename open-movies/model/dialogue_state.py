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
        if hasattr(lf, 'titles') and lf.titles:
            curr_title = None
            for title in lf.titles:
                if not title.canonical.value in self.mentioned_titles:
                    curr_title = title
                    break
            if not curr_title:
                curr_title = lf.titles[0]
            self.curr_title = curr_title.canonical.value
            self.mentioned_titles.add(self.curr_title)

