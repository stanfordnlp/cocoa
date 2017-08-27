import numpy as np
import time
from collections import Counter

class Vocabulary(object):

    UNK = '<unk>'

    def __init__(self, offset=0, unk=True):
        self.word_to_ind = {}
        self.ind_to_word = {}
        self.word_count = Counter()
        self.size = 0
        self.offset = offset
        self.special_words = set()
        if unk:
            self.add_word(self.UNK, special=True)
        self.finished = False

    def add_words(self, words, special=False):
        for w in words:
            self.add_word(w, special)

    def has(self, word):
        return word in self.word_to_ind

    def add_word(self, word, special=False):
        self.word_count[word] += 1
        if special:
            self.special_words.add(word)

    def finish(self, freq_threshold=0, size_threshold=None):
        if freq_threshold > 0:
            for word, count in self.word_count.items():
                if count < freq_threshold:
                    del self.word_count[word]

        self.ind_to_word = [w for w, c in self.word_count.most_common(size_threshold)]
        self.word_to_ind = {w: i for i, w in enumerate(self.ind_to_word)}

        # Make sure special words are included
        n = len(self.ind_to_word)
        for w in self.special_words:
            if w not in self.word_to_ind:
                self.ind_to_word.append(w)
                self.word_to_ind[w] = n
                n += 1

        self.size = len(self.ind_to_word)

        self.finished = True

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

    def load_embeddings(self, wordvec_file, dim):
        print 'Loading pretrained word vectors:', wordvec_file
        start_time = time.time()
        embeddings = np.random.uniform(-1., 1., [self.size, dim])
        num_exist = 0
        with open(wordvec_file, 'r') as f:
            for line in f:
                ss = line.split()
                word = ss[0]
                if word in self.word_to_ind:
                    num_exist += 1
                    vec = np.array([float(x) for x in ss[1:]])
                    embeddings[self.word_to_ind[word]] = vec
        print '[%d s]' % (time.time() - start_time)
        print '%d pretrained' % num_exist
        return embeddings
