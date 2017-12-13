from cocoa.model.dialogue_state import DialogueState as State

class DialogueState(State):
    def __init__(self, agent, kb):
        super(DialogueState, self).__init__(agent, kb)
        self.entities = [[], []]
        self.exclude_entities = [[], []]
        self.mentioned_entities = set()
        self.recent_mentioned_entities = []
        self.matched_item = None
        self.selected_items = []

    @property
    def my_entities(self):
        return self.entities[self.agent]

    @my_entities.setter
    def my_entities(self, entities):
        self.entities[self.agent] = entities

    @property
    def partner_entities(self):
        return self.entities[self.partner]

    @property
    def partner_exclude_entities(self):
        return self.exclude_entities[self.partner]

    def update(self, agent, utterance):
        super(DialogueState, self).update(agent, utterance)
        lf = utterance.lf
        if lf.intent != 'select':
            if hasattr(lf, 'entities'):
                self.entities[agent] = lf.entities
                self.mentioned_entities.update(lf.entities)
                self.recent_mentioned_entities.extend(lf.entities)
                self.recent_mentioned_entities = self.recent_mentioned_entities[-10:]
            if hasattr(lf, 'exclude_entities'):
                self.exclude_entities[agent] = lf.exclude_entities
                self.mentioned_entities.update(lf.exclude_entities)
        else:
            self.selected_items.append(lf.item)
            if lf.matched:
                self.matched_item = lf.item
