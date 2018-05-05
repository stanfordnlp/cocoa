from cocoa.model.vocab import Vocabulary
from cocoa.core.entity import is_entity

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
