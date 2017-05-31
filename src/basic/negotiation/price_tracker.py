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
    def _get_price_range(cls, kb, partner_kb):
        if kb is not None:
            b = kb['personal']['Bottomline']
            #t = kb['personal']['Target']
            p = kb['item']['Price']
            role = kb['personal']['Role']
            if role == 'seller':
                price_range = (b, p)
            else:
                price_range = (0.6*b, b)
        elif partner_kb is not None:
            b = partner_kb['personal']['Bottomline']
            p = partner_kb['item']['Price']
            if partner_kb['personal']['Role'] == 'buyer':
                role = 'seller'
            else:
                role = 'buyer'
            # Approximate range
            if role == 'buyer':
                # b is seller's bottomline, which is higher than buyer's bottomline
                price_range = (0.8*b, p)
            else:
                # b is buyer's bottomline, which is lower than buyer's bottomline
                price_range = (0.6*b, p)
        else:
            raise Exception('No KB is provided')
        return price_range

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
        #price_range = self._get_price_range(kb, partner_kb)
        entity_tokens = []
        N = len(raw_tokens)
        for i, token in enumerate(raw_tokens):
            try:
                number = float(self.process_string(token))
                if i + 1 < N and raw_tokens[i+1].startswith('mile'):
                    new_token = token
                #elif number >= price_range[0] and number <= price_range[1]:
                #    new_token = (token, (number, 'price'))
                else:
                    new_token = token
            except ValueError:
                new_token = token
            entity_tokens.append(new_token)
        return entity_tokens

