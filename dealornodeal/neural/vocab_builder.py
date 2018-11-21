from cocoa.model.vocab import Vocabulary
from cocoa.neural.vocab_builder import build_utterance_vocab

from symbols import markers, sequence_markers

def build_kb_vocab(dialogues, special_symbols=[]):
    kb_vocab = Vocabulary(offset=0, unk=False)
    for dialogue in dialogues:
        assert dialogue.is_int is False
        kb_vocab.add_words(dialogue.scenario)

    kb_vocab.add_words(special_symbols, special=True)
    kb_vocab.finish()

    print 'KB vocab size:', kb_vocab.size
    return kb_vocab

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
    kb_vocab = build_kb_vocab(dialogues)
    return {'utterance_vocab': utterance_vocab,
            'kb_vocab': kb_vocab,
            }
