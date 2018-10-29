import random
import numpy as np
from itertools import izip

from neural.symbols import markers
from core.event import Event
from sessions.rulebased_session import RulebasedSession

class HybridSession(RulebasedSession):
    def receive(self, event):
        if event.action in Event.decorative_events:
            return

        # process the rulebased portion
        utterance = self.parser.parse(event, self.state)
        #print('action fed into neural mananger: {}'.format(utterance.lf))
        self.state.update(self.partner, utterance)

        # process the neural based portion
        if event.action == "message":
            lf_tokens = [utterance.lf.intent]
            if utterance.lf.proposal is not None:
                # Proposal from partner's perspective
                prop = utterance.lf.proposal[1-self.agent]
                #lf_tokens.extend([str(prop[x]) for x in self.manager.items])
                lf_tokens.extend(self.manager._proposal_to_tokens(prop))
            event.data = ' '.join(lf_tokens)
            print 'event.data to manager:', event.data
            self.manager.receive(event)
        else:
            self.manager.receive(event)

    # Generator makes sure that the action is valid
    #def is_valid_action(self, action_tokens):
    #    if not action_tokens:
    #        return False
    #    if action_tokens[0] in self.price_actions and \
    #            not (len(action_tokens) > 1 and is_entity(action_tokens[1])):
    #        return False
    #    return True

    def _get_send_proposal(self, action_tokens):
        proposal_for_me = self.manager._get_proposal(action_tokens)
        proposal_for_partner = {item: self.item_counts[item] - x for item, x in proposal_for_me.iteritems()}
        proposal = {self.agent: proposal_for_me,
                1 - self.agent: proposal_for_partner}
        return proposal

    def send(self):
        action_tokens = self.manager.generate()
        print 'action sent by manager:', action_tokens
        if action_tokens is None:
            return None
        self.manager.dialogue.add_utterance(self.agent, list(action_tokens))

        #price = None
        #if not self.is_valid_action(action_tokens):
        #    action = 'unknown'
        #else:
        action = action_tokens[0]
        if action in ('propose', 'insist', markers.SELECT):
            proposal = self._get_send_proposal(action_tokens)
            if action == markers.SELECT:
                return self.select(proposal[self.agent])
            return self.propose(proposal, action)

        #if action == 'unknown':
        #    return self.propose(self.state.my_proposal)
        if action == markers.QUIT:
            return self.quit()

        return self.template_message(action)
