import re
from cocoa.core.entity import Entity

class Lexicon(object):
    """Detect item and numbers in a list of tokens.

    Example:
        ['i', 'would', 'like', 'the', 'ball', 'and', '2', 'books'] ->
        ['i', 'would', 'like', 'the', ('ball', ('ball', 'item')), 'and', ('2', (2, 'number')), ('books', ('book', 'item'))]

    """

    word_to_num = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}

    def __init__(self, items):
        self.items = items

    def detect_item(self, token):
        for item in self.items:
            if re.match(r'{}s?'.format(item), token) or \
                (item == 'ball' and re.match(r'(basket)?balls?', token)):
                    return Entity.from_elements(surface=token, value=item, type='item')
        return False

    def detect_number(self, token):
        try:
            n = int(token)
        except ValueError:
            try:
                n = self.word_to_num[token]
            except KeyError:
                n = None
        if n is not None:
            return Entity.from_elements(surface=token, value=n, type='number')
        return False

    def link_entity(self, tokens):
        return [(self.detect_item(token) or self.detect_number(token) or token) for token in tokens]

############### TEST ###############
if __name__ == '__main__':
    lexicon = Lexicon(['ball', 'hat', 'book'])
    print lexicon.link_entity('i need 3 books'.split())
