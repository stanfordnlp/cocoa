from cocoa.neural.utterance import Utterance
from cocoa.neural.utterance import UtteranceBuilder as BaseUtteranceBuilder

from symbols import markers
from cocoa.core.entity import is_entity

class UtteranceBuilder(BaseUtteranceBuilder):
    """
    Build a word-based utterance from the batch output
    of generator and the underlying dictionaries.
    """
    def scene_to_sent(self, variables, vocab):
        sent_ids = variables.data.cpu().numpy()
        pad_id = vocab.to_ind(markers.PAD)
        sent_words = [vocab.to_word(x) for x in sent_ids if x != pad_id]

        book = "Book count: {}, value: {}".format(sent_words[0], sent_words[1])
        hat = "Hat count: {}, value: {}".format(sent_words[2], sent_words[3])
        ball = "Ball count: {}, value: {}".format(sent_words[4], sent_words[5])
        return [book, hat, ball]