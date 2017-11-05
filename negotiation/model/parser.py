import re
import numpy as np
from nltk import ngrams
from collections import defaultdict

from cocoa.core.entity import is_entity
from cocoa.model.parser import Parser as BaseParser, LogicalForm, Utterance

from core.tokenizer import tokenize

class Parser(BaseParser):
    price_patterns = [
            r'come down',
            r'(highest|lowest)',
            r'go (lower|higher)',
            r'too (high|low)',
            ]

    greeting_patterns = [
            r'how are you',
            r'interested in',
            ]

    agreement_patterns = [
        r'^that works[.!]*$',
        r'^great[.!]*$',
        r'^(ok|okay)[.!]*$',
        r'^great, thanks[.!]*$',
        r'^deal[.!]*$',
        r'^[\w ]*have a deal[\w ]*$',
        r'^i can do that[.]*$',
    ]

    @classmethod
    def is_greeting(cls, utterance):
        result = super(Parser, cls).is_greeting(utterance)
        if not result:
            for pattern in cls.greeting_patterns:
                if re.search(pattern, utterance.text, re.IGNORECASE):
                    return True
            return False
        return result

    @classmethod
    def is_agreement(cls, utterance):
        for pattern in cls.agreement_patterns:
            if re.match(pattern, utterance.text, re.IGNORECASE) is not None:
                return True
        return False

    @classmethod
    def is_price(cls, utterance):
        for pattern in cls.price_patterns:
            if re.search(pattern, utterance.text):
                return True
        return False

    def compare(self, x, y, inc):
        """Compare two prices.

        Args:
            x (float)
            y (float)
            inc (float={1,-1}): 1 means higher is better and -1 mean lower is better.
        """
        if x == y:
            r = 0
        elif x < y:
            r = -1
        else:
            r = 1
        r *= inc
        return r

    def parse_offer(self, event):
        intent = 'offer'
        try:
            price = float(event.data.get('price'))
        except TypeError:
            price = None
        return LogicalForm(intent, price=price)

    def tag_utterance(self, utterance):
        """Tag the utterance with basic speech acts.
        """
        tags = super(Parser, self).tag_utterance(utterance)
        if self.is_price(utterance):
            tags.append('vague-price')
        if self.is_agreement(utterance):
            tags.append('agree')
        if sum([1 if is_entity(token) else 0 for token in utterance.tokens]) > 0:
            tags.append('price')
        return tags

    def parse_proposed_price(self, prices, dialogue_state):
        """When the partner mentions multiple prices, decide which one they meant.
        """
        partner_prices = []
        listing_price = self.kb.listing_price
        curr_price = dialogue_state['curr_price']
        partner_prev_price = dialogue_state['price'][self.partner]
        partner_role = 'seller' if self.kb.role == 'buyer' else 'buyer'
        inc = 1. if partner_role == 'seller' else -1.
        for price in prices:
            price = price.canonical.value
            # 1) New price doest repeat current price
            # 2) One's latest price is worse than previous ones (compromise)
            # 3) Seller might propose the listing price but the buyer won't
            if price != curr_price and \
                (partner_prev_price is None or self.compare(price, partner_prev_price, inc) <= 0) and \
                (partner_role == 'seller' or price != listing_price):
                    partner_prices.append(price)
        # If more than one possible prices, pick the best one
        if partner_prices:
            i = np.argmax(inc * np.array(partner_prices))
            return partner_prices[i]
        return None

    def parse_message(self, event, dialogue_state):
        tokens = self.lexicon.link_entity(tokenize(event.data), kb=self.kb, scale=False)
        utterance = Utterance(event.data, tokens)
        prices = [token for token in tokens if is_entity(token)]
        proposed_price = self.parse_proposed_price(prices, dialogue_state)
        tags = self.tag_utterance(utterance)
        if dialogue_state['time'] == 1 and not 'price' in tags:
            intent = 'intro'
        elif 'price' in tags and dialogue_state['curr_price'] is None:
            intent = 'init-price'
        elif (not 'price' in tags) and 'vague-price' in tags:
            intent = 'vague-price'
        elif proposed_price is not None:
            intent = 'counter-price'
        elif tags == ['question']:
            intent = 'inquiry'
        elif 'agree' in tags:
            intent = 'agree'
        elif not tags:
            my_prev_act = dialogue_state['act'][self.agent]
            if my_prev_act and my_prev_act.intent == 'inquiry':
                intent = 'inform'
        else:
            intent = 'unknown'
        return LogicalForm(intent, price=proposed_price)

    def parse(self, event, dialogue_state, update_state=False):
        # We are parsing the partner's utterance
        assert event.agent == 1 - self.agent
        if event.action == 'offer':
            lf = self.parse_offer(event)
        elif event.action == 'message':
            lf = self.parse_message(event, dialogue_state)
        else:
            return False

        if update_state:
            dialogue_state['act'][self.partner] = lf
            if lf.price:
                dialogue_state['price'][self.partner] = lf.price
                dialogue_state['curr_price'] = lf.price
        # TODO: insist on price (repeat theri prev price)
        # TODO: agree with price (repeat my priv price)
        #if lf.intent == 'unknown':
        #    print event.data
        return lf

def parse_dialogue(example, lexicon, counter):
    kbs = example.scenario.kbs
    parsers = [Parser(agent, kbs[agent], lexicon) for agent in (0, 1)]
    dialogue_state = {
            'time': 0,
            'act': [None, None],
            'price': [None, None],
            'curr_price': None,
            }
    lfs = []
    for i, event in enumerate(example.events):
        dialogue_state['time'] = i
        lf = parsers[1 - event.agent].parse(event, dialogue_state, update_state=True)
        if lf:
            lfs.append(lf)
    counter['total_lf'] += len(lfs)
    counter['unk_lf'] += len([lf for lf in lfs if lf.intent == 'unknown'])
    if len(lfs) > 2:
        for seq in ngrams(lfs, 3):
            seq = [s.intent for s in seq]
            counter['seqs'][seq[0]][seq[1]][seq[2]] += 1


if __name__ == '__main__':
    import argparse
    from collections import defaultdict
    from cocoa.core.dataset import read_examples
    from core.price_tracker import PriceTracker
    from core.scenario import Scenario
    from model.preprocess import Preprocessor

    parser = argparse.ArgumentParser()
    parser.add_argument('--transcripts', nargs='*', help='JSON transcripts to extract templates')
    parser.add_argument('--price-tracker-model')
    parser.add_argument('--max-examples', default=-1, type=int)
    args = parser.parse_args()

    price_tracker = PriceTracker(args.price_tracker_model)
    examples = read_examples(args.transcripts, args.max_examples, Scenario)
    counter = {'total_lf': 0, 'unk_lf': 0, 'seqs': defaultdict(lambda : defaultdict(lambda : defaultdict(int)))}
    for example in examples:
        if Preprocessor.skip_example(example):
            continue
        parse_dialogue(example, price_tracker, counter)
    print '% unk:', counter['unk_lf'] / float(counter['total_lf'])
    print 'intent seqs:'
    seqs = counter['seqs']
    for first in seqs:
        total_first = sum([sum(seqs[first][a].values()) for a in seqs[first]])
        unks = sum([sum(seqs[first][a].values()) for a in ('unknown',)])
        unks = float(unks) / total_first
        print first, 'unk={:.3f}'.format(unks)
        for second in seqs[first]:
            if second == 'unknown':
                continue
            total = sum(seqs[first][second].values())
            if total < 50:
                continue
            unks = seqs[first][second]['unknown']
            unks = float(unks) / total
            print first, second, '{:.3f}'.format(total/float(total_first)), 'unk={:.3f}'.format(unks)
        print '------------------------'
