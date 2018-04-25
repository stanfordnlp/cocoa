from cocoa.model.vocab import Vocabulary
from cocoa.core.entity import is_entity

from symbols import markers
#from preprocess import Preprocessor

# TODO: import from preprocess (require to separate datagenerator and preprocessor)
def get_entity_form(entity, form):
    assert len(entity) == 2
    if form == 'surface':
        return entity.surface
    elif form == 'type':
        return '<%s>' % entity.canonical.type
    elif form == 'canonical':
        return entity._replace(surface='')
    else:
        raise ValueError('Unknown entity form %s' % form)

def build_utterance_vocab(dialogues, special_symbols=[], entity_forms=[]):
    vocab = Vocabulary(offset=0, unk=True)

    def _add_entity(entity):
        for entity_form in entity_forms:
            word = get_entity_form(entity, entity_form)
            vocab.add_word(word)

    # Add words
    for dialogue in dialogues:
        assert dialogue.is_int is False
        for turn in dialogue.token_turns:
            for token in turn:
                if is_entity(token):
                    _add_entity(token)
                else:
                    vocab.add_word(token)

    # Add special symbols
    vocab.add_words(special_symbols, special=True)
    vocab.finish(size_threshold=10000)
    print 'Utterance vocab size:', vocab.size
    return vocab

def build_kb_vocab(dialogues, special_symbols=[]):
    kb_vocab = Vocabulary(offset=0, unk=True)
    cat_vocab = Vocabulary(offset=0, unk=False)

    for dialogue in dialogues:
        assert dialogue.is_int is False
        kb_vocab.add_words(dialogue.title)
        kb_vocab.add_words(dialogue.description)
        cat_vocab.add_word(dialogue.category)

    kb_vocab.add_words(special_symbols, special=True)
    kb_vocab.finish(freq_threshold=5)
    cat_vocab.add_words(['bike', 'car', 'electronics', 'furniture', 'housing', 'phone'], special=True)
    cat_vocab.finish()

    print 'KB vocab size:', kb_vocab.size
    print 'Category vocab size:', cat_vocab.size
    return kb_vocab, cat_vocab

def build_lf_vocab(dialogues):
    vocab = Vocabulary(offset=0, unk=True)
    for dialogue in dialogues:
        assert dialogue.is_int is False
        for lf in dialogue.lfs:
            vocab.add_words(lf)
    vocab.add_words([markers.GO_S, markers.GO_B, markers.EOS, markers.PAD], special=True)
    vocab.finish()
    print 'LF vocabulary size:', vocab.size
    return vocab

def create_mappings(dialogues, schema, entity_forms):
    utterance_vocab = build_utterance_vocab(dialogues, markers, entity_forms)
    kb_vocab, cat_vocab = build_kb_vocab(dialogues, [markers.PAD])
    lf_vocab = build_lf_vocab(dialogues)
    return {'utterance_vocab': utterance_vocab,
            'kb_vocab': kb_vocab,
            'cat_vocab': cat_vocab,
            'lf_vocab': lf_vocab,
            }
