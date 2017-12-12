import re
import copy

from cocoa.core.entity import is_entity
from cocoa.model.parser import Parser as BaseParser, LogicalForm as LF, Utterance

from core.tokenizer import tokenize

class Parser(BaseParser):
    ME = 0
    YOU = 1

    neg_words = ("nothing", "zero", "dont", "worthless")
    i_words = ('i', 'ill', 'id', 'me', 'mine', 'my')
    you_words = ('u', 'you', 'yours', 'your')
    sentence_delimiter = ('.', ';', '?')

    @classmethod
    def is_negative(cls, utterance):
        neg = super(Parser, cls).is_negative(utterance)
        if neg:
            return True
        for token in utterance.tokens:
            if token in cls.neg_words:
                return True
        return False

    @classmethod
    def is_agree(cls, utterance):
        if re.search(r'ok|okay|deal|sure|fine|yes|yeah|good|work|great|perfect', utterance.text.lower()) and not cls.is_negative(utterance):
            return True
        return False

    def _is_item(self, token):
        return is_entity(token) and token.canonical.type == 'item'

    def _is_number(self, token):
        return is_entity(token) and token.canonical.type == 'number'

    def proposal_to_str(self, proposal, item_counts):
        s = []
        for agent in (self.ME, self.YOU):
            ss = ['me' if agent == self.ME else 'you']
            if agent in proposal:
                p = proposal[agent]
                # TODO: sort items
                for item in ('book', 'hat', 'ball'):
                    count = 'none' if (not item in p) or p[item] == 0 \
                            else 'all' if p[item] == item_counts[item] \
                            else 'number'
                    ss.append(count)
            else:
                ss.extend(['none']*3)
            s.append(','.join(ss))
        #print 'proposal type:', '|'.join(s)
        return '|'.join(s)

    def parse_proposal(self, tokens, item_counts):
        proposal = {self.ME: {}, self.YOU: {}}
        items = []
        curr_agent = None
        uncertain = False

        def pop_items(agent, items):
            if agent is None or len(items) == 0:
                return
            for item, count in items:
                proposal[agent][item] = count
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
                uncertain = True
            pop_items(curr_agent, items)

        if not proposal[self.ME] and not proposal[self.YOU]:
            return None, None, None

        #print 'explict proposal:', proposal
        proposal_type = self.proposal_to_str(proposal, item_counts)

        # Inform: don't need item
        if proposal[self.ME] and not proposal[self.YOU] and sum(proposal[self.ME].values()) == 0:
            for item, count in item_counts.iteritems():
                # Take everything else
                if item not in proposal[self.ME]:
                    proposal[self.ME][item] = count

        # Merge proposal
        proposal = self.merge_proposal(proposal, item_counts, self.ME)
        # proposal: inferred proposal for both agents (after merge)
        # proposal_type: proposal mentioned in the utterance (before merge)
        return proposal, proposal_type, uncertain

    def merge_proposal(self, proposal, item_counts, speaking_agent):
        # Complete proposal
        for agent in proposal:
            if len(proposal[agent]) > 0:
                for item in item_counts:
                    if not item in proposal[agent]:
                        proposal[agent][item] = 0

        partner = 1 - speaking_agent
        for item, count in item_counts.iteritems():
            my_count = proposal[speaking_agent].get(item)
            if my_count is not None:
                proposal[partner][item] = count - my_count
            else:
                partner_count = proposal[partner].get(item)
                if partner_count is not None:
                    proposal[speaking_agent][item] = count - partner_count
                # Should not happend: both are None
                else:
                    print ('WARNING: trying to merge proposals but both counts are none.')
                    proposal[speaking_agent][item] = count
                    proposal[partner][item] = 0
        return proposal

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

        count = min(count, total)
        if count is None:
            return total, True
        else:
            return count, False

    def has_item(self, utterance):
        for token in utterance.tokens:
            if self._is_item(token):
                return True
        return False

    def extract_template(self, tokens, dialogue_state):
        template = []
        for token in tokens:
            if self._is_item(token):
                item = token.canonical.value
                if template and template[-1] == '{number}':
                    template[-1] = '{{{0}-number}}'.format(item)
                template.append('{{{0}}}'.format(item))
            elif self._is_number(token):
                template.append('{number}')
            else:
                template.append(token)
        return template

    def classify_intent(self, utterance):
        if self.has_item(utterance):
            intent = 'propose'
        elif self.is_agree(utterance):
            intent = 'agree'
        elif self.is_negative(utterance):
            intent = 'disagree'
        elif self.is_question(utterance):
            intent = 'inquire'
        elif self.is_greeting(utterance):
            intent = 'greet'
        else:
            intent = 'unknown'
        return intent

    def parse_message(self, event, dialogue_state):
        tokens = self.lexicon.link_entity(tokenize(event.data))
        utterance = Utterance(raw_text=event.data, tokens=tokens)
        intent = self.classify_intent(utterance)

        split = None
        proposal_type = None
        ambiguous_proposal = False
        if intent == 'propose':
            proposal, proposal_type, ambiguous_proposal = self.parse_proposal(utterance.tokens, self.kb.item_counts)
            if proposal:
                # NOTE: YOU/ME in proposal is from the partner's perspective
                split = {self.agent: proposal[self.YOU], self.partner: proposal[self.ME]}
                if dialogue_state.partner_proposal and split[self.partner] == dialogue_state.partner_proposal[self.partner]:
                    intent = 'insist'
        lf = LF(intent, proposal=split, proposal_type=proposal_type)
        utterance.lf = lf

        utterance.template = self.extract_template(tokens, dialogue_state)
        utterance.ambiguous_template = ambiguous_proposal

        return utterance

    def parse(self, event, dialogue_state):
        # We are parsing the partner's utterance
        assert event.agent == 1 - self.agent
        if event.action in ('reject', 'select'):
            u = self.parse_action(event)
        elif event.action == 'message':
            u = self.parse_message(event, dialogue_state)
        else:
            return False

        return u

    def test(self, c, d, raw_utterance, lexicon):
        scenario = {'book':c[0] , 'hat':c[1], 'ball':c[2]}
        proposal, _, _ = self.parse_proposal(lexicon.link_entity(tokenize(raw_utterance)), scenario)
        if not proposal:
            print 'No offer detected:', raw_utterance
            return False

        passed = True
        for i, item in enumerate(('book', 'hat', 'ball')):
            if proposal[self.ME][item] != d[i]:
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
          print proposal[self.ME]
          print 'For you:'
          print proposal[self.YOU]
          print("  The correct proposal is {0} books, {1} hats, and {2} balls".format(d[0], d[1], d[2]) )

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
