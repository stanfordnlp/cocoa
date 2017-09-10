from cocoa.core.kb import KB as BaseKB

class KB(BaseKB):
    def __init__(self, attributes, facts):
        super(KB, self).__init__(attributes)
        self.facts = facts
        # "Attributes" is the schema of attributes
        # Facts are the instiation of those atrributes for a certain agent

    def to_dict(self):
        return self.facts

    @staticmethod
    def from_dict(attributes, raw):
        return KB(attributes, raw)

    def dump(self):
        books, hats, balls = self.facts['Item_counts']['book'], self.facts['Item_counts']['hat'], self.facts['Item_counts']['ball']
        print 'Items Available: {0} books, {1} hats, {2} balls'.format(books, hats, balls)
        book_value = self.facts['Item_values']['book']
        print 'How you value books: {} points'.format(book_value)
        hat_value = self.facts['Item_values']['hat']
        print 'How you value hats: {} points'.format(hat_value)
        ball_value = self.facts['Item_values']['ball']
        print 'How you value balls: {} points'.format(ball_value)
        print '----------------'

        # if self.facts['Strategy'] == 'obsessed':
        #     print 'Since you only care about one item, you are obsessed.'
        # elif self.facts['Strategy'] == 'overvalued':
        #     print 'Since you really only care about two items, those items are overvalued.'
        # elif self.facts['Strategy'] == 'balanced':
        #     print 'Since you care about all items, your view is balanced.'
