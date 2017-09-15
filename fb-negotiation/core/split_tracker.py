import math
import re
import random
from collections import defaultdict
from itertools import chain
from cocoa.core.util import read_json, write_pickle, read_pickle
from tokenizer import tokenize

class SplitTracker(object):
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
          break
        elif token in ["you"]:
          mention['agent'] = 'for_me'
          break
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

  def link_entity(self, raw_tokens, kb=None, scale=True, price_clip=None):
    tokens = ['<s>'] + raw_tokens + ['</s>']
    entity_tokens = []
    if kb:
      kb_numbers = self.get_kb_numbers(kb)
      list_price = kb.facts['item']['Price']
    for i in xrange(1, len(tokens)-1):
      token = tokens[i]
      try:
        number = float(self.process_string(token))
        # Check context
        if not (token[0] == '$' or token[-1] == '$') and \
            not self.is_price(tokens[i-1], tokens[i+1]):
          number = None
        # Avoid 'infinity' being recognized as a number
        if number == float('inf') or number == float('-inf'):
          number = None
        # Check if the price is reasonable
        elif kb:
          if number > 1.5 * list_price:
            number = None
          # Probably a spec number
          if number != list_price and number in kb_numbers:
            number = None
          if number is not None and price_clip is not None:
            scaled_price = PriceScaler._scale_price(kb, number)
            if abs(scaled_price) > price_clip:
              number = None
      except ValueError:
        number = None
      if number is None:
        new_token = token
      else:
        assert not math.isnan(number)
        if scale:
          scaled_price = PriceScaler._scale_price(kb, number)
        else:
          scaled_price = number
        new_token = Entity(surface=token, canonical=CanonicalEntity(value=scaled_price, type='price'))
      entity_tokens.append(new_token)
    return entity_tokens

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--train-examples-path', help='Path to training json file')
  parser.add_argument('--output', help='Path to output model')
  args = parser.parse_args()

  tracker = SplitTracker()
  pass_counter = 0
  total_counter = 0
  with open(args.train_examples_path, 'r') as file:
    for idx, line1 in enumerate(file):
      scenario = [int(x) for x in line1.rstrip().split(" ")]
      line2 = next(file)
      correct = [int(x) for x in line2.rstrip().split(" ")]
      line3 = next(file)
      tracker.reset_tracker()
      if tracker.unit_test(scenario, correct, line3.rstrip()):
        pass_counter += 1
      total_counter += 1
  print("Passed {0} of {1} unit tests.".format(pass_counter, total_counter) )
