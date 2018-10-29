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
        # <pad> token removed from kb_vocab, so no need to check
        # pad_id = vocab.to_ind(markers.PAD)
        sent_words = [vocab.to_word(x) for x in sent_ids]
        title = "KB SCENARIO:"
        book = "  Book count: {}, value: {}".format(sent_words[0], sent_words[1])
        hat = "  Hat count: {}, value: {}".format(sent_words[2], sent_words[3])
        ball = "  Ball count: {}, value: {}".format(sent_words[4], sent_words[5])
        return [title, book, hat, ball]

    def selection_to_sent(self, variables, vocab):
        select_ids = variables.data.cpu().numpy()
        sel = [vocab.to_word(x) for x in select_ids]

        title = "OUTCOME PRED:"
        mine = "  My book: {}, hat: {}, ball {}".format(sel[0], sel[1], sel[2])
        theirs = "  Their book: {}, hat: {}, ball: {}".format(sel[3], sel[4], sel[5])
        return [title, mine, theirs]

    def _entity_to_str(self, entity_token, kb):
        # there is no price scaling here, so we can just return the entity
        return str(entity_token.canonical.value)

