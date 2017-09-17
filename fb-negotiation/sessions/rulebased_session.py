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
    def get_session(agent, kb, tracker, config=None):
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

        self.state = {
                'introduced': False,
                'selected': False,
                'last_act': None,
                'their_action': None,
                'last_utterance': None,
                'num_utterance': 0,
                'last_offer': None
                # 'final_called': False,
                # 'num_partner_insist': 0,
                # 'num_persuade': 0,
                # 'sides': set(),
                }

        self.init_item_ranking()
        self.set_breakpoints()
        self.pick_strategy()
        self.initialize_tracker()

    def pick_strategy(self):
        valuation = [self.item_values[item] for item in self.items]
        # if there is one item valued at 8 or higher, that warrants an obsession
        if max(valuation) >= self.good_deal:
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
            self.state['num_utterance'] += 1

    def init_item_ranking(self):
        values = copy.deepcopy(self.item_values)
        max_key = operator.itemgetter(1)

        self.top_item = max(values.iteritems(), key=max_key)[0]
        values[self.top_item] = -10
        self.middle_item = max(values.iteritems(), key=max_key)[0]
        values[self.middle_item] = -10
        self.bottom_item = max(values.iteritems(), key=max_key)[0]

        if self.item_values[self.middle_item] == self.item_values[self.bottom_item]:
            if self.item_counts[self.bottom_item] > self.item_counts[self.middle_item]:
                self.bottom_item, self.middle_item = self.middle_item, self.bottom_item

    def set_breakpoints(self):
        self.good_deal = 7    # self.kb.facts['personal']['good_deal']
        self.final_call = 2    # minimum number of points willing to accept

        option_A = self.item_values[self.top_item]
        option_B = 2 * self.item_values[self.middle_item]
        option_C = 5 # points
        self.bottomline = min(option_A, option_B, option_C)

    def initialize_tracker(self):
        self.tracker.set_item_counts(self.item_counts)
        self.tracker.reset()

    def process_offer(self):
        offer = self.reverse(self.tracker.their_offer)
        self.state['introduced'] = True

        if self.meets_criteria(offer, 'good_deal'):
            return self.agree()
        elif self.meets_criteria(offer, 'my_proposal'):
            return self.agree()
        elif self.meets_criteria(offer, 'bottomline'):
            return self.negotiate()
        else: # their offer is below the bottomline
            return self.play_hardball()

    def process_persuasion(self):
        if self.state['their_action'] == 'disagree':
            self.state['selected'] = True
            self.my_proposal = self.reverse(self.last_offer)
            return self.message("OK, we can go with what you said earlier then.")
        elif len(self.tracker.lexicon) > 0:
            self.tracker.determine_which_agent()
            self.tracker.resolve_persuasion(self.last_offer)
            self.my_proposal = self.reverse(self.tracker.their_offer)
            return self.agree()
        else:
            return self.clarify()

    def negotiate(self):
        self.state['num_utterance'] += 1
        if self.state['num_utterance'] < 2:
            return self.propose()
        elif self.state['num_utterance'] <= 3:
            return self.persuade()
        elif self.state['num_utterance'] == 4:
            return self.compromise()
        elif self.state['num_utterance'] == 5:
            return self.final_call()
        elif self.state['num_utterance'] >= 6:
            return self.reject()

    def play_hardball(self):
        self.state['num_utterance'] += 1
        if self.state['num_utterance'] < 2:
            s = ["You drive a hard bargain here!",
                "That is too low, I can't do that!",
                "{0}s are worth {1} points to me, I can't take that!".format(
                    self.bottom_item, self.item_values[self.bottom_item])
                ]
            return self.message(random.choice(s))
        elif self.state['num_utterance'] <= 3:
            return self.propose()
        elif self.state['num_utterance'] == 4:
            return self.compromise()
        elif self.state['num_utterance'] == 5:
            return self.final_call()
        elif self.state['num_utterance'] >= 6:
            return self.reject()

    def check_question(self, tokens):
        is_question = False
        if tokens[-1] == "?":
            is_question = True
        elif tokens[0].lower() in ['what', 'which']:
            is_question = True
        elif tokens[1].lower() in ['what', 'which']:
            is_question = True

        if is_question:
            self.state['last_act'] = 'heard_question'

    def check_disagreement(self, tokens):
        for token in tokens:
            if token.lower() in ["nope", "not", "cannot", "can't", "sorry"]:
                self.state['their_action'] = 'disagree'

    def check_agreement(self, raw_utterance, tokens):
        they_agree = False
        regexes = [
          re.compile('(W|w)orks for me'),
          re.compile('(I|i) can (take|do|accept)'),
          re.compile('(S|s)ounds (good|great)'),
        ]
        if any([regex.search(raw_utterance) for regex in regexes]):
            they_agree = True
        if "deal" in tokens:
            they_agree = True
            word_index = tokens.index("deal")
            previous_tokens = tokens[:word_index]
            neg_words = ['no', 'cannot', 'not']
            if any([token in neg_words for token in previous_tokens]):
                they_agree = False
        if they_agree:
            self.finalize_my_proposal()
            self.state['selected'] = True

    def overvalued_proposal(self):
        test_proposal = copy.deepcopy(self.my_proposal)
        test_proposal[self.middle_item] = 1
        if self.meets_criteria(test_proposal, "good_deal"):
            self.my_proposal[self.middle_item] = 1
        else:
            self.my_proposal[self.middle_item] = 2

        prop = self.offer_to_string(self.my_proposal)
        s = ["I would really like " + prop + ".",
            "Would it be ok for me to get " + prop + "?",
            "How about I get " + prop + "?"
        ]
        return self.message(random.choice(s))

    def balanced_proposal(self):
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
        prop = self.offer_to_string(self.my_proposal)
        s = ["I would realy like " + prop + ".",
            "Would if be ok for me to get " + prop + "?",
            "How about I get " + prop + "?"
        ]
        return self.message(random.choice(s))

    def pluralize(self, item, word=None):
        if self.item_counts[item] > 1:
            if word == None:
                return item + "s"
            if word == "is":
                return "are"
            if word == "looks":
                return "look"
        else:
            if word == None:
                return item
            else:
                return word

    def obsessed_proposal(self):
        top = self.pluralize(self.top_item)
        s = ["I would really like the " + top + ", you can have the rest!",
            "The " + top + " " + self.pluralize(self.top_item, "is") + \
                " the only item worth anything to me, you can have the rest!",
            "Hmm, I only really get points for the " + top + "."
            ]
        return self.message(random.choice(s))

    def propose(self):
        self.state['introduced'] = True
        self.state['last_act'] = 'propose'
        self.state['num_utterance'] += 1
        # if I have not yet made a proposal
        if not self.my_proposal['made']:
            return self.init_propose()
        else:
            self.my_proposal['made'] = True
            if self.strategy == 'obsessed':
                return self.obsessed_proposal()
            if self.strategy == 'overvalued':
                return self.overvalued_proposal()
            elif self.strategy == 'balanced':
                return self.balanced_proposal()

    def final_call():
        # If they are only offering 0 or 1 points, then
        # might as well reject since "No Deal" does not cause negative reward
        if self.meets_criteria(self.tracker.their_offer, "final_call"):
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
                "Hi, what would you like?"
            ]
        return self.message(random.choice(s))

    def init_propose(self):
        self.my_proposal['made'] = True
        self.state['introduced'] = True
        self.state['last_act'] = 'init_propose'

        if self.strategy == 'obsessed':
            self.my_proposal[self.top_item] = self.item_counts[self.top_item]
            self.my_proposal[self.middle_item] = 0
            self.my_proposal[self.bottom_item] = 0
            top = self.pluralize(self.top_item)

            s = ["The " + top + " alone looks good to me. What about you?",
                "I only need the " + top + ", you can have the rest!",
                "I would really appreciate getting just the " + top + " :)"
                ]
        elif self.strategy == 'overvalued':
            self.my_proposal[self.top_item] = 1
            self.my_proposal[self.middle_item] = 1
            self.my_proposal[self.bottom_item] = 0
            flipped_offer = self.offer_to_string( self.reverse(self.my_proposal) )

            s = ["What if I get a " + self.top_item + " along with a " + \
                         self. middle_item + " and you take the rest?",
                "I would like " +self.offer_to_string(self.my_proposal)+ "please.",
                "How does " + flipped_offer + "for you sound?",
                ]

        elif self.strategy == 'balanced':
            s = [   "They all look good to me, what do you want?",
                    "Hi, they all look nice, what do you propose?",
                    "Would you like to have all the " + self.bottom_item + "s?",
                ]

        return self.message(random.choice(s))

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
            my_total = sum([self.item_values[item] * self.my_proposal[item] for item in self.items])
            return False if my_total < 0 else (total_points >= my_total)

    def agree(self):
        self.state['selected'] = True
        self.state['last_act'] = 'agree'
        self.finalize_my_proposal()

        s = ["Great deal, thanks!",
          "Yes, that sounds good",
          "Perfect, sounds like we have a deal!",
          "OK, it's a deal"]
        return self.message(random.choice(s))

    def persuade(self):
        self.state['last_act'] = 'persuade'   # 'request_more'
        s = [   "Can you do better than that?",
                "Maybe just one more item?",
                "How about just one more item?"
            ]
        # if self.top_item == 'book':
        #     persuade_detail = [
        #         "I have always been a book worm.",
        #         "The books come in a set, so I would want them all.",
        #         "I'm trying to complete my collection of novels in this series.",
        #         ]
        # elif self.category == 'hat':
        #     persuade_detail = [
        #         "I need to hide a bald spot with the hat.",
        #         "People tell me I look great with a hat on.",
        #         "This hat fits perfectly with my head.",
        #         ]
        # elif self.category == 'ball':
        #     persuade_detail = [
        #         "I have always loved sports.",
        #         "I need these for my youth rec league.",
        #         "You would look great in a hat.",
        #         ]
        return self.message(random.choice(s))

    def reverse(self, offer):
        reverse_offer = {}
        for item in self.items:
            reverse_offer[item] = self.item_counts[item] - offer[item]
        return reverse_offer

    def compromise(self):
        if self.meets_criteria(self.tracker.their_offer, "bottomline"):
            return self.agree()

        package_A = copy.deepcopy(self.tracker.their_offer)
        top_value_item = self.find_high_value(package_A)
        package_A[top_value_item] -= 1
        points_A = self.deal_points(package_A)

        package_B = copy.deepcopy(self.my_proposal)
        low_value_item = self.find_low_value(package_B)
        package_B[low_value_item] -= 1
        points_B = self.deal_points(package_B)

        if points_A < 0:
            package = "B"
        elif points_B < 0:
            package = "A"
        elif points_A < points_B:
            package = "A"
        else:
            package = "B"

        if package == "A":
            s = "How about this, you can have " + self.offer_to_string(package_A)
            self.my_proposal = self.reverse(package_A)
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
        use_no = True if sum([offer[item] == 0 for item in self.items]) < 2 else False

        message_string = ""
        for idx, item in enumerate(self.items):
            offer_count = offer[item]
            if use_no:
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
            else:
                if offer_count > 0:
                    message_string += "{0} {1}s ".format(offer_count, item)
        return message_string

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

    def finalize_my_proposal(self):
        for item in self.items:
            offer = self.tracker.their_offer[item]
            if self.state['last_act'] == 'agree':
                self.my_proposal[item] = self.item_counts[item] - offer
            elif self.my_proposal[item] < 0 and offer >= 0:
                self.my_proposal[item] = self.item_counts[item] - offer
        self.state['last_act'] = 'select'

    def mark_deal_agreed(self):
        outcome = {}
        outcome['deal_points'] = self.deal_points()
        outcome['item_split'] = self.my_proposal

        return self.select(outcome)

    def receive(self, event):
        if event.action == 'select':
            self.state['selected'] = True
            self.state['last_act'] = 'select'
        elif event.action == 'reject':
            self.state['last_act'] = 'reject'
        elif event.action == 'message':
            tokens = tokenize(event.data)
            if self.state['last_act'] == 'persuade':
                self.last_offer = self.tracker.their_offer
                self.check_disagreement(tokens)
            elif len(tokens) == 1:
                self.state['their_action'] = 'one_word_response'
            else:
                self.check_question(tokens)
                self.check_agreement(event.data, tokens)
            self.tracker.reset()
            self.tracker.build_lexicon(tokens)

    def send(self):
        if self.state['selected']:
            if self.state['last_act'] == 'select':
                return self.mark_deal_agreed()
            # The check on deal_points is more of a unit test, rather than
            # to ensure a good deal, since default points are negative.
            if self.deal_points() >= 0:
                self.finalize_my_proposal()
                return self.mark_deal_agreed()
                # self.message("Wonderful, we have a deal!")
            else:
                return self.reject()

        if self.state['last_act'] == 'reject':
            return self.reject()
        if self.state['last_act'] == 'persuade':
            self.tracker.determine_item_count()
            return self.process_persuasion()

        if self.tracker.made['their_offer']:
            self.state['their_action'] = 'propose'
            self.tracker.determine_item_count()
            self.tracker.determine_which_agent()
            # print("A {}".format(self.tracker.lexicon) )
            self.tracker.resolve_tracker()
            # print("B {}".format(self.tracker.their_offer) )
            self.tracker.merge_their_offers()
            # print("C {}".format(self.tracker.their_offer) )
            return self.process_offer()

        if self.state['last_act'] == 'heard_question':
            return self.propose()

        if not self.state['introduced']:
            if random.random() < 0.5:   # talk a bit by asking a question
                return self.intro()     # to hear their side first
            elif not self.my_proposal['made']:    # make a light proposal
                return self.init_propose()      # to get the ball rolling

        if not self.tracker.made['their_offer']:
            if self.state['last_act'] == 'init_propose':
                return self.propose()
            else:
                return self.init_propose()

        if self.tracker.needs_clarification:
            return self.clarify()
        if self.state['their_action'] == 'one_word_response':
            return self.one_word_clarify()

        raise Exception('Uncaught case')