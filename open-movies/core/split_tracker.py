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

        def pop_items(agent, items):
            for item, count in items:
                split[curr_agent][item] = count
            del items[:]

        # TODO: pop items before switching curr_agent
        for i, token in enumerate(tokens):
            if token in ('i', 'ill', 'id', 'me'):
                curr_agent = agent
            elif token in ('u', 'you'):
                curr_agent = 1 - agent
            elif is_entity(token):
                if token.canonical.type == 'item':
                    item = token.canonical.value
                    count, clarify = self.parse_count(token, tokens[i-1] if i > 0 else None, item_counts[item])
                    need_clarify = need_clarify or clarify
                    items.append((item, count))
            if curr_agent != -1 and len(items) > 0:
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

    def parse_count(self, token, prev_token, total):
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
        elif prev_token == 'no':
            return 0, False
        elif prev_token in ('the', 'all'):
            return total, False
        else:
            return max(1, total / 2), True

    def unit_test(self, c, d, raw_utterance):
        scenario = {'book':c[0] , 'hat':c[1], 'ball':c[2]}
        agent = 0
        split = self.parse_offer(tokenize(raw_utterance), agent, scenario)
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
