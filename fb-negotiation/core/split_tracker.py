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
                    count = self.parse_count(token, tokens[i-1] if i > 0 else None, item_counts[item])
                    items.append((item, count))
            if curr_agent != -1 and len(items) > 0:
                pop_items(curr_agent, items)
        # Clean up. Assuming it's for the speaking agent if no subject is mentioned.
        if len(items) > 0:
            if curr_agent == -1:
                curr_agent = agent
            pop_items(curr_agent, items)

        if len(split) == 0:
            return None
        else:
            return self.merge_offers(split, item_counts, agent)

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
                    print ('WARNINT: trying to merge offers but both counts are none.')
                    split[me][item] = total
                    split[them][item] = 0
        return dict(split)

    def parse_count(self, token, prev_token, total):
        if prev_token is None:
            return total
        elif is_entity(prev_token) and prev_token.canonical.type == 'number':
            return min(prev_token.canonical.value, total)
        elif prev_token in ('a', 'an'):
            return 1
        elif prev_token == 'no':
            return 0
        else:
            return total

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

class SplitTracker_(object):
  def __init__(self):
    self.made = {'their_offer': False, 'their_offer_for_me': False}
    self.their_offer = {'book':-1, 'hat':-1, 'ball':-1}
    self.their_offer_for_me = {'book':-1, 'hat':-1, 'ball':-1}
    self.items = ['book', 'hat', 'ball']
    self.item_counts = None
    self.needs_clarification = False
    self.lexicon = []

  def set_item_counts(self, item_counts):
    self.item_counts = item_counts

  def previous_words(self, tokens):
    for mention in self.lexicon:
      item_idx = mention['location']
      two_words_back = item_idx - 2
      one_word_back = item_idx - 1

      mention['two_back'] = tokens[two_words_back] if two_words_back >= 0 else None
      mention['one_back'] = tokens[one_word_back] if one_word_back >= 0 else None  # penultimate

  def before_and_after(self, tokens):
    for mention in self.lexicon:
      item_idx = mention['location']
      mention['before'] = reversed(tokens[:item_idx])
      mention['after'] = tokens[item_idx+1:]

  def reset(self):
    self.made = {'their_offer': False, 'their_offer_for_me': False}
    self.their_offer = {'book':-1, 'hat':-1, 'ball':-1}
    self.their_offer_for_me = {'book':-1, 'hat':-1, 'ball':-1}
    self.lexicon = []  # clear out previous lexicon

  def build_lexicon(self, tokens):
    for idx, token in enumerate(tokens):
      if re.match('books?', token):
        self.lexicon.append( {"item_type": 'book', "location": idx} )
      elif re.match('hats?', token):
        self.lexicon.append( {"item_type": 'hat', "location": idx} )
      elif re.match('(basket)?balls?', token):
        self.lexicon.append( {"item_type": 'ball', "location": idx} )
      elif token in ['rest', 'everything']:
        self.lexicon.append( {"item_type": 'rest', "location": idx} )
      elif token == 'each':
        self.lexicon.append( {"item_type": 'each', "location": idx} )

    if len(self.lexicon) > 0:
      self.made['their_offer'] = True
      self.previous_words(tokens)
      self.before_and_after(tokens)

  def check_missing(self):
    missing_items = 0
    captured_items = [mention['item_type'] for mention in self.lexicon]

    for item in self.items:
      if item not in captured_items:
        missing_items += 1

    if missing_items == 0:
      self.determine_item_count()
    elif missing_items == 1:
      self.clarify()
    if missing_items >= 2:
      pass # they are probably asking a question

  def determine_item_count(self):
    for mention in self.lexicon:
      if mention['one_back'] in ['one', 'a', 'an', '1']:
        mention['count'] = 1
      elif mention['one_back'] in ['two', '2', 'some']:
        mention['count'] = 2
      elif mention['one_back'] in ['three', '3']:
        mention['count'] = 3
      elif mention['one_back'] in ['four', '4']:
        mention['count'] = 4
      elif mention['one_back'] in ['zero', '0', 'none', 'no']:
        mention['count'] = 0
      elif mention['one_back'] == 'all':
        mention['count'] = -1
      elif mention['one_back'] == 'the':
        if mention['two_back'] == 'split':
          mention['count'] = -2
        elif mention['item_type'] == 'rest':
          mention['count'] = -3
        else:  # all the balls, both the hats
          mention['count'] = -1
      elif mention['item_type'] == 'rest':
        mention['count'] = -3
      elif mention['item_type'] == 'each':
        mention['count'] = -4
      else:
        mention['count'] = -1

  def resolve_persuasion(self, last_offer):
    if len(self.lexicon) == 1:
      mention = self.lexicon[0]
      last_offer[mention['item_type']] -= 1
      self.their_offer = last_offer
    else:
      self.resolve_tracker()
      self.merge_their_offers()

  def resolve_tracker(self):
    for mention in self.lexicon:
      item_type = mention['item_type']
      count = mention['count']
      if item_type in self.items:
        item_count = self.item_counts[item_type]

      if count >= 0:
        if mention['agent'] == 'for_them':
          self.their_offer[item_type] = count
        elif mention['agent'] == 'for_me':
          self.their_offer_for_me[item_type] = count
          self.made['their_offer_for_me'] = True
      elif count == -1: # all available items
        if mention['agent'] == 'for_them':
          self.their_offer[item_type] = item_count
        elif mention['agent'] == 'for_me':
          self.their_offer_for_me[item_type] = item_count
          self.made['their_offer_for_me'] = True
      elif count == -2: # split item evenly
        if (item_count % 2 == 0):
          self.their_offer[item_type] = item_count/2
        else:
          item_count += 1  # we give them the extra one
          self.their_offer[item_type] = item_count/2
      elif count == -3: # all the rest
        self.resolve_the_rest(mention)
      elif count == -4:
        self.resolve_each(mention)
      else:
        self.needs_clarification = True

  def resolve_the_rest(self, mention):
    if mention['agent'] == 'for_them':
      for item in self.items:
        if self.their_offer_for_me[item] == None:
          self.their_offer[item] = self.item_counts[item]
    elif mention['agent'] == 'for_me':
      for item in self.items: # whatever they haven't claimed is offered to me
        if self.their_offer[item] == None:
          self.their_offer_for_me[item] = self.item_counts[item]

  def resolve_each(self, mention):
    if mention['agent'] == 'for_them':
      if (mention['one_back'] == 'one') or (mention['two_back'] == 'one'):
        for item in self.items:
          self.their_offer[item] = 1
      elif (mention['one_back'] == 'two') or (mention['two_back'] == 'two'):
        for item in self.items:
          self.their_offer[item] = 2
      else:
        self.needs_clarification = True
    elif mention['agent'] == 'for_me':
      if (mention['one_back'] == 'one') or (mention['two_back'] == 'one'):
        self.made['their_offer_for_me'] = True
        for item in self.items:
          self.their_offer_for_me[item] = 1
      elif (mention['one_back'] == 'two') or (mention['two_back'] == 'two'):
        self.made['their_offer_for_me'] = True
        for item in self.items:
          self.their_offer_for_me[item] = 2
      else:
        self.needs_clarification = True

  def determine_which_agent(self):
    for mention in self.lexicon:
      agent_found = False
      for token in mention['before']:
        if token in ["i", "i'd", "i'll"]:
          mention['agent'] = 'for_them'
          agent_found = True
          break
        elif token in ["you", "u"]:
          mention['agent'] = 'for_me'
          agent_found = True
          break
      if agent_found:
        continue
      for token in mention['after']:
        if token == "me":
          mention['agent'] = 'for_them'
          agent_found = True
          break
        elif token in ["you"]:
          mention['agent'] = 'for_me'
          agent_found = True
          break
      if agent_found:
        continue
      # by default, if nothing matches, we assume the offer was for them
      mention['agent'] = 'for_them'

  def merge_their_offers(self):
    for item in self.items:
      if self.made['their_offer_for_me'] == True:
        # if they already stated what they want, then do not mess with it
        if self.their_offer[item] >= 0:
          for_them = self.their_offer[item]
        # they should get the opposite of whatever they believe I should have
        elif self.their_offer[item] < 0 and self.their_offer_for_me[item] >= 0:
          for_them = self.item_counts[item] - self.their_offer_for_me[item]
        # if they proposed something, assume anything not mentioned is for them
        else:
          for_them = self.item_counts[item]
        self.their_offer[item] = for_them
      elif self.their_offer[item] < 0:
        self.their_offer[item] = 0

  def unit_test(self, c, d, raw_utterance):
    scenario = {'book':c[0] , 'hat':c[1], 'ball':c[2]}
    self.set_item_counts(scenario)

    self.build_lexicon(tokenize(raw_utterance))
    self.determine_item_count()
    self.determine_which_agent()
    self.resolve_tracker()
    self.merge_their_offers()

    book_match = self.their_offer['book'] == d[0]
    hat_match = self.their_offer['hat'] == d[1]
    ball_match = self.their_offer['ball'] == d[2]

    if book_match and hat_match and ball_match:
      print("Passed")
      passed = True
    else:
      print("TEST SCENARIO")
      print("  There are {0} books, {1} hats, and {2} balls.".format(c[0], c[1], c[2]) )
      print("  Sentence: {0}".format(raw_utterance) )
      print("SYSTEM OUTPUT")
      print("  They want {0} books, {1} hats, and {2} balls".format(
          self.their_offer['book'], self.their_offer['hat'], self.their_offer['ball']))
      for item in self.items:
        ct = self.their_offer_for_me[item]
        if ct > 0:
          print("  They think I should should get {0} {1}s".format(ct, item))
      print("  The correct split is {0} books, {1} hats, and {2} balls".format(d[0], d[1], d[2]) )
      print("LEXICON")
      for mention in self.lexicon:
        print mention
      passed = False

    print("------------------------------")
    return passed

  @staticmethod
  def clarify():
    print("Ask for clarification")
    # If you heard no items:
    #   check for question mark

if __name__ == '__main__':
  import argparse
  from lexicon import Lexicon
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--train-examples-path', help='Path to training json file')
  args = parser.parse_args()

  lexicon = Lexicon(['ball', 'hat', 'book'])
  tracker = SplitTracker(lexicon)
  pass_counter = 0
  total_counter = 0
  with open(args.train_examples_path, 'r') as file:
    for idx, line1 in enumerate(file):
        scenario = [int(x) for x in line1.rstrip().split(" ")]
        line2 = next(file)
        correct = [int(x) for x in line2.rstrip().split(" ")]
        line3 = next(file)
        #tracker.reset()
        if tracker.unit_test(scenario, correct, line3.rstrip()):
          pass_counter += 1
        total_counter += 1
  print("Passed {0} of {1} unit tests.".format(pass_counter, total_counter) )
