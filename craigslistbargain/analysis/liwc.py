import re
from cocoa.core.util import read_pickle, write_pickle
import pygtrie

class LIWC(object):
    def __init__(self, category_map, prefix_trie, words):
        self.category_map = category_map
        self.words = words
        self.prefix_trie = prefix_trie
        self.category_subset = set(category_map.keys())  # ints

    def _map_categories(self, categories):
        return [self.category_map[c] for c in categories if c in self.category_subset]

    def lookup(self, word):
        categories = self.words.get(word, None)
        if categories is not None:
            #print 'word'
            return self._map_categories(categories)
        _, categories = self.prefix_trie.longest_prefix(word)
        #print 'prefix'
        if not categories:
            return []
        return self._map_categories(categories)

    @classmethod
    def from_pkl(cls, path):
        d = read_pickle(path)
        return cls(d['category_map'], d['prefix_trie'], d['words'])

    @classmethod
    def from_txt(cls, path, output=None):
        category_map = {}
        prefix_trie = pygtrie.CharTrie()
        words = {}
        with open(path, 'r') as fin:
            in_cat = None
            for line in fin:
                line = line.strip()
                if line == '%':
                    if in_cat is None:
                        in_cat = True
                    elif in_cat:
                        in_cat = False
                elif in_cat:
                    ss = line.split('\t')
                    category_map[int(ss[0])] = ss[1]
                else:
                    ss = line.split('\t')
                    categories = []
                    for x in ss[1:]:
                        try:
                            cat_id = int(x)
                            categories.append(cat_id)
                        except ValueError:
                            continue
                    if '*' in ss[0]:
                        assert ss[0][-1] == '*'
                        prefix = ss[0].replace('*', '')
                        prefix_trie[prefix] = categories
                    else:
                        words[ss[0]] = categories
        d = {'category_map': category_map, 'prefix_trie': prefix_trie, 'words': words}
        if output:
            write_pickle(d, output)
        return cls(category_map, prefix_trie, words)

if __name__ == '__main__':
    liwc = LIWC.from_txt('data/LIWC2007_English080730.dic', output='data/liwc.pkl')
    print liwc.lookup("alot")
