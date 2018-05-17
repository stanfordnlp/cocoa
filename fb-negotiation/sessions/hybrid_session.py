import random
import numpy as np
import pdb
from core.event import Event
from sessions.rulebased_session import RulebasedSession

class HybridSession(RulebasedSession):
    def receive(self, event):
        if event.action in Event.decorative_events:
            return
        # process the rulebased portion
        utterance = self.parser.parse(event, self.state)
        print('action fed into neural mananger: {}'.format(utterance.lf))
        self.state.update(self.partner, utterance)
        # process the neural based portion
        if event.action == "message":
            logical_form = {"intent": utterance.lf.intent, "price": utterance.lf.price}
            entity_tokens = self.manager.env.preprocessor.lf_to_tokens(self.kb, logical_form)
        else:
            logical_form = None
            entity_tokens = self.manager.env.preprocessor.process_event(event, self.kb)
        if entity_tokens:
            self.manager.dialogue.add_utterance(event.agent, entity_tokens, logical_form)

    # called by the send() method of the parent rulebased session
    def choose_action(self):
        self.manager.dialogue.is_int = False
        action = self.manager.generate()[0]
        print("action predicted by neural manager: {}".format(action))
        p_act = self.state.partner_act
        if action == "unknown" and (p_act == "accept" or p_act == "agree"):
            action = "agree"

        return action if action else 'unknown'

