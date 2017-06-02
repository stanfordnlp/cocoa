from src.basic.entity import Entity, CanonicalEntity
import re

class PriceTracker(object):
    @classmethod
    def get_price(cls, token):
        try:
            return token.canonical.value
        except:
            try:
                return token.value
            except:
                return None

    @classmethod
    def process_string(cls, token):
        token = re.sub(r'[\$\,]', '', token)
        try:
            if token.endswith('k'):
                token = str(float(token.replace('k', '')) * 1000)
        except ValueError:
            pass
        return token

    def link_entity(self, raw_tokens, kb=None, partner_kb=None, mentioned_entities=None):
        '''
        Detect numbers:
            ['how', 'about', '1000'] => ['how', 'about', ('1000', (1000, 'price'))]
        '''
        # We must know at least one KB
        assert kb or partner_kb
        entity_tokens = []
        N = len(raw_tokens)
        for i, token in enumerate(raw_tokens):
            try:
                number = float(self.process_string(token))
                if i + 1 < N and (\
                        raw_tokens[i+1].startswith('mile') or\
                        raw_tokens[i+1].startswith('year')\
                        ):
                    new_token = token
                else:
                    new_token = Entity(surface=token, canonical=CanonicalEntity(value=number, type='price'))
            except ValueError:
                new_token = token
            entity_tokens.append(new_token)
        return entity_tokens

