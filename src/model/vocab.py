# TODO: use named tuple to represent entities?
def is_entity(word):
    if not isinstance(word, basestring):
        return True
    return False

class Vocabulary(object):

    UNK = '<unk>'

    def __init__(self, offset=0, unk=True):
        self.word_to_ind = {}
        self.ind_to_word = {}
        self.size = 0
        self.offset = offset
        if unk:
            self.add_word(self.UNK)

    def add_words(self, words):
        for w in words:
            self.add_word(w)

    def has(self, word):
        return word in self.word_to_ind

    def add_word(self, word):
        if not self.has(word):
            ind = self.size + self.offset
            self.word_to_ind[word] = ind
            self.ind_to_word[ind] = word
            self.size += 1

    def to_ind(self, word):
        if word in self.word_to_ind:
            return self.word_to_ind[word]
        else:
            # NOTE: if UNK is not enabled, it will throw an exception
            if self.UNK in self.word_to_ind:
                return self.word_to_ind[self.UNK]
            else:
                raise KeyError(str(word))

    def to_word(self, ind):
        return self.ind_to_word[ind]

    def dump(self):
        for i, w in self.ind_to_word.iteritems():
            print '{:<8}{:<}'.format(i, w)
