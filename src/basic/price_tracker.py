# TODO: combine this with the Lexicon

class PriceTracker(object):
    @classmethod
    def _get_price_range(cls, kb, partner_kb):
        if kb is not None:
            b = kb['personal']['Bottomline']
            t = kb['personal']['Target']
        elif partner_kb is not None:
            b = partner_kb['personal']['Bottomline']
            t = partner_kb['personal']['Target']
            # Approximate range
            if partner_kb['personal']['Role'] == 'buyer':
                # [target, bottomline]
                b = b + (b - t)
            else:
                # [bottomline, target]
                b = max(0, b - (t - b))
        else:
            raise Exception('No KB is provided')
        if b < t:
            price_range = (b, t)
        else:
            price_range = (t, b)
        return price_range

    @classmethod
    def process_string(cls, token):
        token = token.replace('$', '')
        token = token.replace(',', '')
        token = token.replace('K', '000')
        return token

    def link_entity(self, raw_tokens, kb=None, partner_kb=None):
        '''
        Detect numbers:
            ['how', 'about', '1000'] => ['how', 'about', ('1000', (1000, 'price'))]
        '''
        # We must know at least one KB
        assert kb or partner_kb
        price_range = self._get_price_range(kb, partner_kb)
        entity_tokens = []
        N = len(raw_tokens)
        for i, token in enumerate(raw_tokens):
            try:
                number = float(self.process_string(token))
                if i + 1 < N and raw_tokens[i+1].startswith('mile'):
                    new_token = token
                elif number >= price_range[0] and number <= price_range[1]:
                    new_token = (token, (number, 'price'))
                else:
                    new_token = token
            except ValueError:
                new_token = token
            entity_tokens.append(new_token)
        return entity_tokens

