import math
import re
import random
from collections import defaultdict
from itertools import chain

from cocoa.core.util import read_json, write_pickle, read_pickle
from cocoa.core.entity import Entity, is_entity

from tokenizer import tokenize

class SplitTracker(object):
    def parse_offer(self, tokens, agent, item_counts):
        split = defaultdict(dict)
        items = []
        curr_agent = -1
        need_clarify = False
        neg_words = ("no", "not", "nothing", "n't", "zero", "dont")
        sentence_delimiter = ('.', ',', ';')

        def pop_items(agent, items):
            if agent == -1 or len(items) == 0:
                return
            for item, count in items:
                split[curr_agent][item] = count
            del items[:]

        neg = False
        for i, token in enumerate(tokens):
            if token in ('i', 'ill', 'id', 'me'):
                pop_items(curr_agent, items)
                curr_agent = agent
            elif token in ('u', 'you'):
                pop_items(curr_agent, items)
                curr_agent = 1 - agent
            elif token in neg_words:
                neg = True
            elif token in sentence_delimiter:
                neg = False
                curr_agent = -1
            elif is_entity(token):
                if token.canonical.type == 'item':
                    item = token.canonical.value
                    count, clarify = self.parse_count(token, tokens[i-1] if i > 0 else None, item_counts[item], neg)
                    need_clarify = need_clarify or clarify
                    items.append((item, count))
            pop_items(curr_agent, items)
        # Clean up. Assuming it's for the speaking agent if no subject is mentioned.
        if len(items) > 0:
            if curr_agent == -1:
                curr_agent = agent
            pop_items(curr_agent, items)

        if len(split) == 0:
            return None, None
        else:
            return self.merge_offers(split, item_counts, agent), need_clarify

    def merge_offers(self, split, item_counts, speaking_agent):
        # If a proposal exists, assume non-mentioned item counts to be zero.
        for agent in (0, 1):
            if agent in split:
                for item in item_counts:
                    if not item in split[agent]:
                        split[agent][item] = 0

        me = speaking_agent
        them = 1 - speaking_agent
        for item, count in item_counts.iteritems():
            my_count = split[me].get(item)
            if my_count is not None:
                split[them][item] = count - my_count
            else:
                their_count = split[them].get(item)
                if their_count is not None:
                    split[me][item] = count - their_count
                # Should not happend: both are None
                else:
                    print ('WARNING: trying to merge offers but both counts are none.')
                    split[me][item] = total
                    split[them][item] = 0
        return dict(split)

    def parse_count(self, token, prev_token, total, negative):
        """
        Returns:
            count (int), clarify (bool)
        """
        if prev_token is None:
            return total, True
        elif is_entity(prev_token) and prev_token.canonical.type == 'number':
            return min(prev_token.canonical.value, total), False
        elif prev_token in ('a', 'an'):
            return 1, False
        elif prev_token == 'both':
            return 2, False
        elif prev_token == 'no':
            return 0, False
        elif prev_token in ('the', 'all'):
            return total, False
        elif negative:
            return 0, False
        else:
            return total, True

    def unit_test(self, c, d, raw_utterance, lexicon):
        scenario = {'book':c[0] , 'hat':c[1], 'ball':c[2]}
        agent = 0
        split, _ = self.parse_offer(lexicon.link_entity(tokenize(raw_utterance)), agent, scenario)
        if not split:
            print 'No offer detected:', raw_utterance
            return False

        book_match = split[agent]['book'] == d[0]
        hat_match = split[agent]['hat'] == d[1]
        ball_match = split[agent]['ball'] == d[2]

        if book_match and hat_match and ball_match:
          print("Passed")
          passed = True
        else:
          print("TEST SCENARIO")
          print("  There are {0} books, {1} hats, and {2} balls.".format(c[0], c[1], c[2]) )
          print("  Sentence: {0}".format(raw_utterance) )
          print("SYSTEM OUTPUT")
          print("  They want {0} books, {1} hats, and {2} balls".format(
              split[agent]['book'], split[agent]['hat'], split[agent]['ball']))
          for item in scenario:
            ct = split[1 - agent][item]
            if ct > 0:
              print("  They think I should should get {0} {1}s".format(ct, item))
          print("  The correct split is {0} books, {1} hats, and {2} balls".format(d[0], d[1], d[2]) )
          passed = False

        print("------------------------------")
        return passed

if __name__ == '__main__':
  import argparse
  from core.lexicon import Lexicon
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--train-examples-path', help='Path to training json file')
  parser.add_argument('--output', help='Path to output model')
  args = parser.parse_args()

  lexicon = Lexicon(['ball', 'hat', 'book'])
  tracker = SplitTracker()
  pass_counter = 0
  total_counter = 0
  with open(args.train_examples_path, 'r') as file:
    for idx, line1 in enumerate(file):
        scenario = [int(x) for x in line1.rstrip().split(" ")]
        line2 = next(file)
        correct = [int(x) for x in line2.rstrip().split(" ")]
        line3 = next(file)
        if tracker.unit_test(scenario, correct, line3.rstrip(), lexicon):
          pass_counter += 1
        total_counter += 1
  print("Passed {0} of {1} unit tests.".format(pass_counter, total_counter) )
