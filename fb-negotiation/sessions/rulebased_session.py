import random
from cocoa.core.entity import is_entity
from session import Session
from core.tokenizer import tokenize
import copy
import sys
import re
import operator

class RulebasedSession(object):
    @staticmethod
    def get_session(agent, kb, tracker=None):
        return BaseRulebasedSession(agent, kb, tracker)

class BaseRulebasedSession(Session):
    def __init__(self, agent, kb, tracker):
        super(BaseRulebasedSession, self).__init__(agent)
        self.agent = agent
        self.kb = kb
        self.tracker = tracker

        self.item_values = kb.facts['Item_values']
        self.item_counts = kb.facts['Item_counts']
        self.items = kb.facts['Item_values'].keys()
        self.my_proposal = {'made': False, 'book':-1, 'hat':-1, 'ball':-1}

        self.pick_strategy()
        self.init_item_ranking()
        self.set_breakpoints()
        self.fill_tracker()

        self.state = {
                'introduced': False,
                'selected': False,
                'last_act': None,
                'their_action': None,
                'last_utterance': None,
                'num_utterance': 0
                # 'has_spoken': False,
                # 'final_called': False,
                # 'num_partner_insist': 0,
                # 'num_persuade': 0,
                # 'sides': set(),
                }

    def pick_strategy(self):
        valuation = copy.deepcopy(self.item_values)
        # if there is one item valued at 8 or higher, that warrants an obsession
        if max(valuation) >= 8:
            self.strategy = "obsessed"
        elif 0 in valuation:
            zero_location = valuation.index(0)
            valuation.pop(zero_location)
            # if there is another 0 in the valuation, then obsessed on that item
            if 0 in valuation:
                self.strategy = "obsessed"
            else:
                self.strategy = "overvalued"
        # if there are no 0-valued items in the set
        else:
            self.strategy = "balanced"

    def init_item_ranking(self):
        values = copy.deepcopy(self.item_values)
        max_key = operator.itemgetter(1)

        self.top_item = max(values.iteritems(), key=max_key)[0]
        values[self.top_item] = -10
        self.middle_item = max(values.iteritems(), key=max_key)[0]
        values[self.middle_item] = -10
        self.bottom_item = max(values.iteritems(), key=max_key)[0]

    def set_breakpoints(self):
        self.good_deal = 8    # self.kb.facts['personal']['good_deal']
        self.final_call = 2    # minimum number of points willing to accept

        option_A = self.item_values[self.top_item]
        option_B = 2 * self.item_values[self.middle_item]
        option_C = 5 # points
        self.bottomline = min(option_A, option_B, option_C)

    def fill_tracker(self):
        self.tracker.set_item_counts(self.item_counts)
        self.latest_offer = None

    def process_offer(self):
        offer = self.tracker.their_offer
        self.state['introduced'] = True
        self.latest_offer = offer

        if self.meets_criteria(offer, 'good_deal'):
            self.state['selected'] = True
            return self.agree()
        elif self.meets_criteria(offer, 'my_proposal'):
            self.state['selected'] = True
            return self.agree()
        elif self.meets_criteria(offer, 'bottomline'):
            return self.negotiate()
        else: # their offer is below the bottomline
            return self.play_hardball()

    def negotiate(self):
        self.state['num_utterance'] += 1
        if self.state['num_utterance'] <= 2:
            return self.propose()
        elif self.state['num_utterance'] == 3:
            return self.persuade()
        elif self.state['num_utterance'] == 4:
            return self.compromise()
        elif self.state['num_utterance'] == 5:
            return self.final_call()
        elif self.state['num_utterance'] >= 6:
            return self.reject()

    def play_hardball(self):
        self.state['num_utterance'] += 1
        if self.state['num_utterance'] <= 2:
            s = ["You drive a hard bargain here!",
                "That is too low, I can't do that!",
                "{0} are worth {1} points to me, I can't take that!".format(
                    self.bottom_item, self.item_values[self.bottom_item])
                ]
            return self.message(random.choice(s))
        elif self.state['num_utterance'] == 3:
            return self.propose()
        elif self.state['num_utterance'] == 4:
            return self.compromise()
        elif self.state['num_utterance'] == 5:
            return self.final_call()
        elif self.state['num_utterance'] >= 6:
            return self.reject()

    def check_agreement(self, raw_utterance):
        regexes = [
          re.compile('(D|d)eal'),
          re.compile('I can (take|do|accept)'),
          re.compile('(S|s)ounds (good|great)'),
        ]

        if any([regex.search(raw_utterance) for regex in regexes]):
          self.mark_deal_agreed()

    def propose(self):
        self.state['introduced'] = True
        self.state['last_act'] = 'propose'
        self.state['num_utterance'] += 1
        # if I have not yet made a proposal
        if not self.my_proposal['made']:
            s = [self.init_propose()]
        else:
            self.state['last_act'] = 'propose'
            if self.strategy == 'obsessed':
                self.my_proposal[self.top_item] = 1
                self.my_proposal['made'] = True
                s = "How about I take the " + self.top_item + " and you take the rest?"
            elif self.strategy == 'overvalued':
                self.my_proposal[self.top_item] = 1
                self.my_proposal[self.middle_item] = 1
                self.my_proposal['made'] = True
                s = "What if I get the " + self.top_item + " along with a " + \
                             self. middle_item + "and you take the rest?"
            elif self.strategy == 'balanced':
                s = [   "They all look good to me, what do you want?",
                        "Hi, that " + self.top_item + " looks really nice.",
                        "Would you like to have all the " + self.bottom_item + "?",
                        "The " + self.top_item + " looks good to me. What about you?"
                    ]


        return self.message(random.choice(s))

    def final_call():
        # If they are only offering 0 or 1 points, then
        # might as well reject since "No Deal" does not cause negative reward
        if meets_criteria(self.tracker.their_offer, "final_call"):
            self.agree()
        else:
            s = ["No, I can't do that.",
                    "Sorry, need more than that",
                    "Let's try something else"]
            return self.message(random.choice(s))

    def valid_proposal(self, offer):
        for item in self.items:
            if offer[item] > self.item_counts[item]:
                return (False, item)
        # if all items pass, then we have valid proposal
        return (True,)

    def intro(self):
        self.state['last_act'] = 'intro'
        self.state['introduced'] = True

        s = [  "So what looks good to you?",
                "Which items do you value highly?",
                "Hi, would what you like?"
            ]
        return self.message(random.choice(s))


    def p_str(self, item):
        if item == 'top':
            proposal_string = str(self.my_proposal[self.top_item])
        elif item == 'mid':
            proposal_string = str(self.my_proposal[self.middle_item])
        elif item == 'btm':
            proposal_string = str(self.my_proposal[self.bottom_item])
        return proposal_string + " "

    def init_propose(self):
        self.my_proposal['made'] = True
        self.my_proposal[self.top_item] = 1
        self.my_proposal[self.middle_item] = 0
        self.my_proposal[self.bottom_item] = 0
        self.state['introduced'] = True
        self.state['last_act'] = 'init_propose'

        if self.strategy == 'overvalued':
            test_proposal = copy.deepcopy(self.my_proposal)
            test_proposal[self.middle_item] += 1
            if self.meets_criteria(test_proposal, "good_deal"):
                self.my_proposal[self.middle_item] += 1
            else:
                self.my_proposal[self.middle_item] += 2

        elif self.strategy == 'balanced':
            test_proposal = copy.deepcopy(self.my_proposal)
            test_proposal[self.middle_item] += 1
            if self.meets_criteria(test_proposal, "good_deal"):
                self.my_proposal[self.middle_item] += 1
            else:
                test_proposal[self.bottom_item] += 1
                if self.meets_criteria(test_proposal, "good_deal"):
                    self.my_proposal[self.middle_item] += 1
                    self.my_proposal[self.bottom_item] += 1
                else:
                    test_proposal[self.top_item] += 1
                    if self.valid_proposal(test_proposal)[0]:
                        self.my_proposal[self.top_item] += 1
                        self.my_proposal[self.middle_item] += 1
                        self.my_proposal[self.bottom_item] += 1
                    else:
                        self.my_proposal[self.middle_item] += 2
                        self.my_proposal[self.bottom_item] += 1

        s = "How about " + self.p_str('top') + str(self.top_item) + ", " \
                         + self.p_str('mid') + str(self.middle_item) + " and " \
                         + self.p_str('btm') + str(self.bottom_item) + "?"
        return self.message(s)

    def meets_criteria(self, offer, deal_type):
        book_total = self.item_values['book'] * offer['book']
        hat_total = self.item_values['hat'] * offer['hat']
        ball_total = self.item_values['ball'] * offer['ball']
        total_points = book_total + hat_total + ball_total

        if deal_type == "good_deal":
            return total_points >= self.good_deal
        elif deal_type == "bottomline":
            return total_points >= self.bottomline
        elif deal_type == "final_call":
            return total_points >= self.final_call
        elif deal_type == "my_proposal":
            my_book = self.item_values['book'] * self.my_proposal['book']
            my_hat = self.item_values['hat'] * self.my_proposal['hat']
            my_ball = self.item_values['ball'] * self.my_proposal['ball']
            my_total_points = my_book + my_hat + my_ball

            return total_points >= my_total_points

    def agree(self):
        s = ["ok deal, thanks!",
          "yes, that sounds good",
          "perfect, sounds like we have a deal",
          "ok, it's a deal"]
        utterance = random.choice(s)
        self.state['last_utterance'] = utterance
        return self.message(utterance)

    def persuade(self):
        if self.top_item == 'book':
            persuade_detail = [
                "I have always been a book worm.",
                "The books come in a set, so I would want them all.",
                "I'm trying to complete my collection of novels in this series.",
                ]
        elif self.category == 'hat':
            persuade_detail = [
                "I need to hide a bald spot with the hat.",
                "People tell me I look great with a hat on.",
                "This hat fits perfectly with my head.",
                ]
        elif self.category == 'ball':
            persuade_detail = [
                "I have always loved sports.",
                "I need these for my youth rec league.",
                "You would look great in a hat.",
                ]
        return self.message(random.choice(persuade_detail))

    def transfer(self, offer):
        for item in self.items:
            self.my_proposal[item] = self.item_values[item] - offer[item]

    def compromise(self):
        package_A = copy.deepcopy(self.tracker.their_offer)
        top_value_item = self.find_high_value(package_A)
        package_A[top_value_item] -= 1
        points_A = self.deal_points(package_A)

        package_B = copy.deepcopy(self.my_proposal)
        low_value_item = self.find_low_value(package_B)
        package_B[low_value_item] -= 1
        points_B = self.deal_points(package_B)

        if points_A < points_B:
            s = "How about this, you can have " + self.offer_to_string(package_A)
            self.transfer(package_A)
        else:
            s = "Hmm, how about I take just " + self.offer_to_string(package_B)
            self.my_proposal = package_B

        return self.message(s)

    def find_high_value(self, package):
        if package[self.top_item] > 0:
            return self.top_item
        elif package[self.middle_item] > 0:
            return self.middle_item
        else:
            return self.bottom_item

    def offer_to_string(self, offer):
        message_string = ""
        for idx, item in enumerate(self.items):
            offer_count = offer[item]
            if offer_count < 1:
                offer_count = "no"
                offer_str = item + "s"
            elif offer_count == 1:
                offer_count = str(offer_count)
                offer_str = item
            elif offer_count > 1:
                offer_count = str(offer_count)
                offer_str = item + "s"

            if idx == 2:
                message_string += "and "
            message_string += offer_count + " " + offer_str + " "

    def find_low_value(self, package):
        if package[self.bottom_item] > 0:
            return self.bottom_item
        elif package[self.middle_item] > 0:
            return self.middle_item
        else:
            return self.top_item

    def clarify(self):
        has_some_idea = False
        for item in self.items:
            if self.tracker.their_offer[item] > 0:
                has_some_idea = True

        if has_some_idea:
            s = ["I believe you want", "I think you want", "Do you want"]
            msg = random.choice(s) + self.offer_to_string(self.tracker.their_offer)
            self.state['last_act'] = 'clarification'
            return self.message(msg)
        else:
            s = ["I'm not sure what you meant there, can you clarify?",
                    "Can you please explain again?",
                    "Sorry, what is it that you want exactly?"
                ]
            return self.message(random.choice(s))

    def one_word_clarify(self):
        s = ["OK, what did you have in mind?",
                "So, what do you think?",
                "So, what are you thinking?"
            ]
        return self.message(random.choice(s))

    def verify_deal(self):
        matches = 0
        for item in self.items:
            offer = self.item_counts[item] - self.tracker.their_offer[item]
            if self.my_proposal[item] == offer:
                matches += 1
        if matches >= 3:
            return True
        else:
            return False

    def deal_points(self, proposal=None):
        if proposal == None:
            proposal = self.my_proposal
        book_total = self.item_values['book'] * proposal['book']
        hat_total = self.item_values['hat'] * proposal['hat']
        ball_total = self.item_values['ball'] * proposal['ball']
        deal_points = book_total + hat_total + ball_total
        return deal_points

    def receive(self, event):
        if event.action == 'message':
            tokens = tokenize(event.data)
            self.tracker.reset()
            self.tracker.build_lexicon(tokens)
            if len(tokens) == 1:
                self.state['their_action'] = 'one_word_response'
            else:
                self.check_agreement(event.data)

        if event.action == 'select':
            self.state['selected'] = True
        if event.action == 'reject':
            self.state['last_act'] = 'reject'

    def mark_deal_agreed(self):
        self.state['selected'] = True

        for item in self.items:
          offer = self.tracker.their_offer[item]
          if self.my_proposal[item] < 0 and offer >= 0:
            self.transfer(self.tracker.their_offer)

        outcome = {}
        outcome['deal_points'] = self.deal_points()
        outcome['item_split'] = self.my_proposal

        return self.select(outcome)

    def send(self):
        if self.state['selected']:
            # The check on deal_points is more of a unit test, rather than
            # to ensure a good deal, since default points are negative.
            if self.deal_points() >= 0:
                return self.mark_deal_agreed()
            else:
                return self.reject()

        if self.state['last_act'] == 'reject':
            return self.reject()

        if self.tracker.made['their_offer']:
            self.state['their_action'] = 'propose'
            self.tracker.determine_item_count()
            self.tracker.determine_which_agent()
            self.tracker.resolve_tracker()
            self.tracker.merge_their_offers()
            return self.process_offer()

        if not self.state['introduced']:
            if random.random() < 0.5:   # talk a bit by asking a question
                return self.intro()     # to hear their side first
            elif not self.my_proposal['made']:    # make a light proposal
                return self.init_propose()      # to get the ball rolling

        if not self.their_offer['made']:
            if self.state['last_act'] == 'init_propose':
                return self.propose()
            else:
                return self.init_propose()

        if self.tracker.needs_clarification:
            return self.clarify()
        if self.state['their_action'] == 'one_word_response':
            return self.one_word_clarify()

        raise Exception('Uncaught case')