class Vocabulary(object):

    UNK = 'UNK'

    def __init__(self):
        self.word_to_ind = {}
        self.size = 0
        self._add_word(self.UNK)

    def add_words(self, words):
        '''
        Add a single word or a list of words to the vocab
        '''
        if isinstance(words, basestring):
            self._add_word(words)
        else:
            for w in words:
                self._add_word(w)

    def _add_word(self, word):
        if word not in self.word_to_ind:
            self.word_to_ind[word] = self.size
            self.size += 1

    def build(self):
        self.ind_to_word = {i: w for w, i in self.word_to_ind.iteritems()}
        print 'Vocabulary size: %d' % (self.size)

    def to_ind(self, word):
        if word in self.word_to_ind:
            return self.word_to_ind[word]
        else:
            return self.word_to_ind[self.UNK]

    def to_word(self, ind):
        return self.ind_to_word[ind]

