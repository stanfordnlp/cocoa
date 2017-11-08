import re

from cocoa.core.entity import is_entity
from cocoa.model.parser import Parser as BaseParser, LogicalForm, Utterance

from core.tokenizer import tokenize

class Parser(BaseParser):
    ME = 0
    YOU = 1

    neg_words = ("no", "not", "nothing", "n't", "zero", "dont", "worthless")
    i_words = ('i', 'ill', 'id', 'me')
    you_words = ('u', 'you')
    sentence_delimiter = ('.', ',', ';')

    @classmethod
    def is_negative(cls, utterance):
        for token in utterance.tokens:
            if token in cls.neg_words:
                return True
        return False

    @classmethod
    def is_agree(cls, utterance):
        if re.search(r'ok|okay|deal|fine|yes|yeah|good|work|great|perfect', utterance.text.lower()) and not cls.is_negative(utterance):
            return True
        return False

    def _is_item(self, token):
        return is_entity(token) and token.canonical.type == 'item'

    def _is_number(self, token):
        return is_entity(token) and token.canonical.type == 'number'

    def parse_offer(self, tokens, item_counts):
        split = {self.ME: {}, self.YOU: {}}
        items = []
        curr_agent = None
        uncertain = False

        def pop_items(agent, items):
            if agent is None or len(items) == 0:
                return
            for item, count in items:
                split[agent][item] = count
            del items[:]

        for i, token in enumerate(tokens):
            pop_items(curr_agent, items)
            if token in self.i_words:
                curr_agent = self.ME
            elif token in self.you_words:
                curr_agent = self.YOU
            # Reset
            elif token in self.sentence_delimiter:
                curr_agent = None
            elif self._is_item(token):
                item = token.canonical.value
                count, guess = self.parse_count(tokens, i, item_counts[item])
                uncertain = uncertain or guess
                items.append((item, count))
        # Clean up. Assuming it's for 'me' if no subject is mentioned.
        if len(items) > 0:
            if curr_agent is None:
                curr_agent = self.ME
            pop_items(curr_agent, items)

        if not split[self.ME] and not split[self.YOU]:
            return None

        # Inform: don't need item
        if split[self.ME] and not split[self.YOU] and sum(split[self.ME].values()) == 0:
            for item, count in item_counts.iteritems():
                if item not in split[self.ME]:
                    split[self.ME][item] = count

        # Complete split
        for agent in (self.ME, self.YOU):
            if split[agent]:
                for item in item_counts:
                    if not item in split[agent]:
                        split[agent][item] = 0

        # Merge split
        for item, count in item_counts.iteritems():
            my_count = split[self.ME].get(item)
            if my_count is not None:
                split[self.YOU][item] = count - my_count
            else:
                your_count = split[self.YOU].get(item)
                if your_count is not None:
                    split[self.ME][item] = count - your_count
                # Should not happend: both are None
                else:
                    print ('WARNING: trying to merge offers but both counts are none.')
                    split[self.ME][item] = count
                    split[self.YOU][item] = 0
        return split

    def parse_count(self, tokens, i, total):
        """Parse count of an item at index `i`.

        Args:
            tokens: all tokens in the utterance
            i (int): position of the item token
            total (int): total count of the item

        Returns:
            count (int)
            guess (bool): True if we are uncertain about the parse

        """
        count = None
        # Search backward
        for j in xrange(i-1, -1, -1):
            token = tokens[j]
            if count is not None or token in self.sentence_delimiter or self._is_item(token):
                break
            elif self._is_number(token):
                count = min(token.canonical.value, total)
            elif token in self.neg_words:
                count = 0
            elif token in ('a', 'an'):
                count =  1
            elif token == 'both':
                count =  2
            elif token in ('the', 'all'):
                count = total

        if count is None:
            # Search forward
            for j in xrange(i+1, len(tokens)):
                token = tokens[j]
                if count is not None or token in self.sentence_delimiter or self._is_item(token):
                    break
                elif count in self.neg_words:
                    count = 0

        if count is None:
            return total, True
        else:
            return count, False

    def has_item(self, utterance):
        for token in utterance.tokens:
            if self._is_item(token):
                return True
        return False

    def parse(self, event, dialogue_state, update_state=False):
        if event.action in ('reject', 'select'):
            lf = LogicalForm(event.action)
        elif event.action == 'message':
            tokens = self.lexicon.link_entity(tokenize(event.data))
            utterance = Utterance(event.data, tokens)
            if self.has_item(utterance):
                split = self.parse_offer(utterance.tokens, self.kb.item_counts)
                if split:
                    # NOTE: YOU/ME in split is from the partner's perspective
                    offer = {self.agent: split[self.YOU], self.partner: split[self.ME]}
                    lf = LogicalForm('propose', offer=offer)
                else:
                    lf = LogicalForm('item')
            elif self.is_agree(utterance):
                lf = LogicalForm('agree')
            elif self.is_negative(utterance):
                lf = LogicalForm('disagree')
            else:
                lf = LogicalForm('unknown')
        else:
            return False

        if update_state:
            dialogue_state['time'] += 1
            dialogue_state['act'][self.partner] = lf
            if lf.intent == 'propose':
                dialogue_state['proposal'][self.partner] = lf.offer

        return lf

    def test(self, c, d, raw_utterance, lexicon):
        scenario = {'book':c[0] , 'hat':c[1], 'ball':c[2]}
        split = self.parse_offer(lexicon.link_entity(tokenize(raw_utterance)), scenario)
        if not split:
            print 'No offer detected:', raw_utterance
            return False

        passed = True
        for i, item in enumerate(('book', 'hat', 'ball')):
            if split[self.ME][item] != d[i]:
                passed = False
                break

        if passed:
          print("Passed")
        else:
          print("TEST SCENARIO")
          print("  There are {0} books, {1} hats, and {2} balls.".format(c[0], c[1], c[2]) )
          print("  Sentence: {0}".format(raw_utterance) )
          print("SYSTEM OUTPUT")
          print 'For me:'
          print split[self.ME]
          print 'For you:'
          print split[self.YOU]
          print("  The correct split is {0} books, {1} hats, and {2} balls".format(d[0], d[1], d[2]) )

        print("------------------------------")
        return passed

if __name__ == '__main__':
    import argparse
    from core.lexicon import Lexicon
    from core.kb import KB
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train-examples-path', help='Path to training json file')
    args = parser.parse_args()

    lexicon = Lexicon(['ball', 'hat', 'book'])
    pass_counter = 0
    total_counter = 0
    with open(args.train_examples_path, 'r') as file:
        for idx, line1 in enumerate(file):
            scenario = [int(x) for x in line1.rstrip().split(" ")]
            parser = Parser(0, None, lexicon)
            line2 = next(file)
            correct = [int(x) for x in line2.rstrip().split(" ")]
            line3 = next(file)
            if parser.test(scenario, correct, line3.rstrip(), lexicon):
              pass_counter += 1
            total_counter += 1
    print("Passed {0} of {1} unit tests.".format(pass_counter, total_counter) )
