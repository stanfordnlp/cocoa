from src.basic.entity import Entity, CanonicalEntity
import re

class PriceScaler(object):
    @classmethod
    def get_price_range(cls, kb):
        '''
        Return the bottomline and the target
        '''
        b = kb.facts['personal']['Bottomline']  # 0
        t = kb.facts['personal']['Target']  # 1
        role = kb.facts['personal']['Role']

        # TODO: should have only one case after we fix the scenarios in the end.
        if b is None:
            if role == 'seller':
                b = t * 0.7
            else:
                b = kb.facts['item']['Price']
        elif t is None:
            if role == 'seller':
                t = kb.facts['item']['Price']
            else:
                t = b * 0.7

        return b, t

    @classmethod
    def get_parameters(cls, b, t):
        '''
        Return (slope, constant) parameters of the linear mapping.
        '''
        assert (t - b) != 0
        w = 1. / (t - b)
        c = -1. * b / (t - b)
        return w, c

    @classmethod
    # TODO: this is operated on canonical entities, need to be consistent!
    def unscale_price(cls, kb, price):
        p = PriceTracker.get_price(price)
        b, t = cls.get_price_range(kb)
        w, c = cls.get_parameters(b, t)
        assert w != 0
        p = (p - c) / w
        p = int(p)
        if isinstance(price, Entity):
            return price._replace(canonical=price.canonical._replace(value=p))
        else:
            return price._replace(value=p)

    @classmethod
    def _scale_price(cls, kb, p):
        b, t = cls.get_price_range(kb)
        w, c = cls.get_parameters(b, t)
        p = w * p + c
        # Discretize to two digits
        p = float('{:.2f}'.format(p))
        return p

    @classmethod
    def scale_price(cls, kb, price):
        '''
        Scale the price such that bottomline=0 and target=1.
        '''
        p = PriceTracker.get_price(price)
        p = cls._scale_price(kb, p)
        return price._replace(canonical=price.canonical._replace(value=p))

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
                scaled_price = PriceScaler._scale_price(kb or partner_kb, number)
                # NOTE: this should capture most non-price numbers
                if scaled_price > 2 or scaled_price < -1:
                    new_token = token
                elif i + 1 < N and (\
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

