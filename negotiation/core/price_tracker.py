import math
import re
from collections import defaultdict
from itertools import chain
from cocoa.core.entity import Entity, CanonicalEntity
from cocoa.core.util import read_json, write_pickle, read_pickle
from tokenizer import tokenize

def add_price_tracker_arguments(parser):
    parser.add_argument('--price-tracker-model', help='Path to price tracker model')

class PriceScaler(object):
    @classmethod
    def get_price_range(cls, kb):
        '''
        Return the bottomline and the target
        '''
        t = kb.facts['personal']['Target']  # 1
        role = kb.facts['personal']['Role']

        if role == 'seller':
            b = t * 0.7
        else:
            b = kb.facts['item']['Price']

        return b, t

    @classmethod
    def get_parameters(cls, b, t):
        '''
        Return (slope, constant) parameters of the linear mapping.
        '''
        assert (t - b) != 0
        w = 1. / (t - b)
        c = -1. * b / (t - b)
        return w, c

    @classmethod
    # TODO: this is operated on canonical entities, need to be consistent!
    def unscale_price(cls, kb, price):
        p = PriceTracker.get_price(price)
        b, t = cls.get_price_range(kb)
        w, c = cls.get_parameters(b, t)
        assert w != 0
        p = (p - c) / w
        p = int(p)
        if isinstance(price, Entity):
            return price._replace(canonical=price.canonical._replace(value=p))
        else:
            return price._replace(value=p)

    @classmethod
    def _scale_price(cls, kb, p):
        b, t = cls.get_price_range(kb)
        w, c = cls.get_parameters(b, t)
        p = w * p + c
        # Discretize to two digits
        p = float('{:.2f}'.format(p))
        return p

    @classmethod
    def scale_price(cls, kb, price):
        '''
        Scale the price such that bottomline=0 and target=1.
        '''
        p = PriceTracker.get_price(price)
        p = cls._scale_price(kb, p)
        return price._replace(canonical=price.canonical._replace(value=p))

class PriceTracker(object):
    def __init__(self, model_path):
        self.model = read_pickle(model_path)

    @classmethod
    def get_price(cls, token):
        try:
            return token.canonical.value
        except:
            try:
                return token.value
            except:
                return None

    @classmethod
    def process_string(cls, token):
        token = re.sub(r'[\$\,]', '', token)
        try:
            if token.endswith('k'):
                token = str(float(token.replace('k', '')) * 1000)
        except ValueError:
            pass
        return token

    def is_price(self, left_context, right_context):
        if left_context in self.model['left'] and right_context in self.model['right']:
            return True
        else:
            return False

    def get_kb_numbers(self, kb):
        title = tokenize(re.sub(r'[^\w0-9\.,]', ' ', kb.facts['item']['Title']))
        description = tokenize(re.sub(r'[^\w0-9\.,]', ' ', ' '.join(kb.facts['item']['Description'])))
        numbers = set()
        for token in chain(title, description):
            try:
                numbers.add(float(self.process_string(token)))
            except ValueError:
                continue
        return numbers

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
                has_dollar = lambda token: token[0] == '$' or token[-1] == '$'
                # Check context
                if not has_dollar(token) and \
                        not self.is_price(tokens[i-1], tokens[i+1]):
                    number = None
                # Avoid 'infinity' being recognized as a number
                elif number == float('inf') or number == float('-inf'):
                    number = None
                # Check if the price is reasonable
                elif kb:
                    if not has_dollar(token):
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

    @classmethod
    def train(cls, examples, output_path=None):
        '''
        examples: json chats
        Use "$xxx$ as ground truth, and record n-gram context before and after the price.
        '''
        context = {'left': defaultdict(int), 'right': defaultdict(int)}
        for ex in examples:
            for event in ex['events']:
                if event['action'] == 'message':
                    tokens = tokenize(event['data'])
                    tokens = ['<s>'] + tokens + ['</s>']
                    for i, token in enumerate(tokens):
                        if token[0] == '$' or token[-1] == '$':
                            context['left'][tokens[i-1]] += 1
                            context['right'][tokens[i+1]] += 1
        if output_path:
            write_pickle(context, output_path)
        return context

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-examples-path', help='Path to training json file')
    parser.add_argument('--output', help='Path to output model')
    args = parser.parse_args()

    examples = read_json(args.train_examples_path)
    PriceTracker.train(examples, args.output)
