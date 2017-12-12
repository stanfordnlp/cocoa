import random
import copy
import sys
import re
import operator
from collections import namedtuple, defaultdict

from cocoa.core.entity import is_entity
from cocoa.model.parser import LogicalForm as LF
from cocoa.sessions.rulebased_session import RulebasedSession as BaseRulebasedSession

from session import Session
from core.tokenizer import tokenize
from model.parser import Parser, Utterance
from model.dialogue_state import DialogueState

Config = namedtuple('Config', ['target', 'bottomline'])
default_config = Config(8, 4)

class RulebasedSession(BaseRulebasedSession):
    def __init__(self, agent, kb, lexicon, config, generator, manager):
        parser = Parser(agent, kb, lexicon)
        state = DialogueState(agent, kb)
        super(RulebasedSession, self).__init__(agent, kb, parser, generator, manager, state, sample_temperature=5.)

        self.kb = kb
        self.item_values = kb.item_values
        self.item_counts = kb.item_counts
        self.items = kb.item_values.keys()
        self.partner_item_weights = {item: 1. for item in self.items}
        self.config = default_config if config is None else config

        items = [(item, value, self.item_counts[item]) for item, value in self.item_values.iteritems()]
        # Sort items by value from high to low
        self.sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        self.init_proposal()

    def init_proposal(self):
        """Initialize my proposal before the conversation begins.

        If there are 0-value items, take all non-zero ones. Otherwise, take
        `target` points.

        """
        # Has zero-value item
        if self.sorted_items[-1][1] == 0:
            my_split = {item: count for item, value, count in self.sorted_items if value > 0}
        # All items have values
        else:
            points = 0
            target = self.config.target
            my_split = {item: 0 for item in self.items}
            for item, value, count in self.sorted_items:
                for i in xrange(count):
                    if points >= target:
                        break
                    my_split[item] += 1
                    points += value
        split = {self.agent: my_split, 1 - self.agent: {}}
        self.state.my_proposal = self.parser.merge_proposal(split, self.item_counts, self.agent)

    def get_points(self, offer):
        return sum([count * self.item_values[item] for item, count in offer.iteritems()])

    def compromise(self, my_offer):
        compromised_offer = copy.deepcopy(my_offer)
        partner_item_values = self.partner_item_weights
        #item_scores = [(item, self.item_values[item] / (partner_item_values[item] if partner_item_values[item] != 0 else 0.5))
        item_scores = [(item, self.item_values[item] - partner_item_values[item])
                for item, count in my_offer.iteritems()
                if count > 0]
        item_scores = sorted(item_scores, key=lambda x: x[1])
        compromised_item = item_scores[0][0]
        compromised_offer[compromised_item] -= 1

        # Take what they don't want
        for item, weight in self.partner_item_weights.iteritems():
            if self.item_values[item] > 0 and weight < 0 and item != compromised_item:
                compromised_offer[item] = self.kb.item_counts[item]

        return compromised_offer

    def negotiate(self):
        if not self.state.partner_proposal:
            partner_proposal = {self.partner: dict(self.item_counts), self.agent: {item: 0 for item in self.item_counts}}
        else:
            partner_proposal = self.state.partner_proposal
        # What I want
        my_offer = self.state.my_proposal[self.agent]

        my_points = self.get_points(my_offer)
        my_points_by_them = self.get_points(partner_proposal[self.agent])

        if (my_points_by_them >= min(self.config.target, my_points) and my_points_by_them <= 10):
            self.state.my_proposal = copy.deepcopy(self.state.partner_proposal)
            return self.clarify()

        compromised_offer = self.compromise(my_offer)
        compromised_points = self.get_points(compromised_offer)
        if compromised_points <= my_points_by_them:
            compromised_offer = copy.deepcopy(self.state.partner_proposal[self.agent])
            compromised_points = my_points_by_them

        if self.get_points(compromised_offer) < self.config.bottomline:
            return self.propose(self.state.my_proposal, 'insist')
        else:
            self.state.my_proposal[self.agent] = compromised_offer
            self.state.my_proposal = self.parser.merge_proposal(self.state.my_proposal, self.item_counts, self.agent)
            # Take partner's offer
            if compromised_points == my_points_by_them:
                return self.clarify()
            return self.propose(self.state.my_proposal)

    def fill_proposal_template(self, template, proposal):
        """
        NOTE: currently only supports templates for my split.
        """
        proposal = proposal[self.parser.ME]
        kwargs = {}
        for item, count in proposal.iteritems():
            kwargs[item] = '{}s'.format(item) if count > 1 else item
            kwargs['{}-number'.format(item)] = count
        s = template.format(**kwargs)
        return s

    def template_message(self, intent):
        template = self.retrieve_response_template(intent)
        lf = LF(intent)
        text = template['template']
        utterance = Utterance(raw_text=text, logical_form=lf, template=template)
        return self.message(utterance)

    def propose(self, proposal, intent='propose'):
        proposal_for_me = {self.parser.ME: proposal[self.agent]}
        proposal_type = self.parser.proposal_to_str(proposal_for_me, self.item_counts)
        template = self.retrieve_response_template(intent, proposal_type=proposal_type)
        if template is None:
            template = "i would like {my_offer}."
            text = self.fill_handcoded_template(template, proposal)
        else:
            text = self.fill_proposal_template(template['template'], proposal_for_me)
        lf = LF(intent, proposal=proposal)
        utterance = Utterance(raw_text=text, template=template, logical_form=lf)
        return self.message(utterance)

    def fill_handcoded_template(self, template, proposal):
        s = template.format(
                my_offer=self.offer_to_string(proposal[self.agent]),
                their_offer=self.offer_to_string(proposal[self.partner])
            )
        return s

    def offer_to_string(self, offer):
        items = ['{count} {item}{plural}'.format(item=item, count=count, plural='s' if count > 1 else '')
                for item, count in offer.iteritems() if count > 0]
        return ' and '.join(items)

    def clarify(self):
        lf = LF('clarify', proposal=self.state.my_proposal)
        s = [
            "so i get {my_offer}, right?",
            "so you get {their_offer}?",
            "ok, i'll take {my_offer}.",
            ]
        template = random.choice(s)
        text = self.fill_handcoded_template(template, self.state.my_proposal)
        utterance = Utterance(raw_text=text, logical_form=lf, template=template)
        return self.message(utterance)

    def receive(self, event):
        super(RulebasedSession, self).receive(event)
        if self.state.partner_proposal is not None:
            for item, count in self.state.partner_proposal[self.partner].iteritems():
                self.partner_item_weights[item] += (1 if count > 0 else -1)

    def wait(self):
        return None

    def select(self, split):
        utterance = Utterance(logical_form=LF('select', split=split))
        self.state.update(self.agent, utterance)
        metadata = self.metadata(utterance)
        return super(RulebasedSession, self).select(split, metadata=metadata)

    def send(self):
        if self.state.partner_act == 'reject':
            return self.reject()

        if self.has_done('select'):
            return self.wait()

        action = self.manager.choose_action(state=self.state)
        if not action:
            action = self.retrieve_action()
            if not action in self.manager.available_actions(self.state):
                action = 'unknown'
        if action in ('propose', 'agree'):
            if not self.has_done('propose'):
                return self.propose(self.state.my_proposal)
            else:
                return self.negotiate()
        elif action == 'select':
            return self.select(self.state.curr_proposal[self.agent])
        elif action == 'clarify':
            return self.clarify()
        elif action == 'unknown':
            return self.propose(self.state.my_proposal)
        else:
            return self.template_message(action)

        raise Exception('Uncaught case')
