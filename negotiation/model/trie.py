import pygtrie
from cocoa.core.util import write_pickle, read_pickle, read_json
from cocoa.core.dataset import Example

class Trie(object):
    def __init__(self, max_prefix_len=10):
        self.max_prefix_len = max_prefix_len
        self.trie = None

    def get_children(self, prefix=()):
        node, _ = self.trie._get_node(prefix)
        return node.children.keys()

    def build_trie(self, seq_iter):
        trie = pygtrie.Trie()
        N = self.max_prefix_len
        for seq in seq_iter:
            for start in xrange(len(seq)):
                key = tuple(seq[start:start+N])
                trie[key] = 1
        self.trie = trie
        return trie
