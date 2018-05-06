from cocoa.model.vocab import Vocabulary
from cocoa.neural.vocab_builder import build_utterance_vocab

from symbols import markers, sequence_markers, item_markers

def build_kb_vocab(select_symbol):
    '''
    Note: For any given item, the max count is 4 and the max value is 10
    With the max point available to any one player being 15
    We increase this to 20 for a small margin of safety
    '''
    kb_vocab = Vocabulary(offset=0, unk=True)
    kb_vocab.add_words(['book', 'hat', 'ball', select_symbol])
    kb_vocab.add_words([str(num) for num in range(20)])
    kb_vocab.add_words(sequence_markers, special=True)
    kb_vocab.finish(freq_threshold=5)

    print 'KB vocab size:', kb_vocab.size
    return kb_vocab

def build_category_vocab():
    cat_vocab = Vocabulary(offset=0, unk=False)
    # cat_vocab.add_word(dialogue.category)
    cat_vocab.add_words(item_markers, special=True)
    cat_vocab.finish()

    print 'Category vocab size:', cat_vocab.size
    return cat_vocab

def build_lf_vocab(dialogues):
    vocab = Vocabulary(offset=0, unk=True)
    for dialogue in dialogues:
        assert dialogue.is_int is False
        for lf in dialogue.lfs:
            vocab.add_words(lf)
    vocab.add_words(sequence_markers, special=True)
    vocab.finish()
    print 'LF vocabulary size:', vocab.size
    return vocab

def create_mappings(dialogues, schema, entity_forms):
    utterance_vocab = build_utterance_vocab(dialogues, sequence_markers, entity_forms)
    kb_vocab = build_kb_vocab(markers.SELECT)
    cat_vocab = build_category_vocab()
    # lf_vocab = build_lf_vocab(dialogues)
    return {'utterance_vocab': utterance_vocab,
            'kb_vocab': kb_vocab,
            'cat_vocab': cat_vocab
            }
            # 'lf_vocab': lf_vocab,
