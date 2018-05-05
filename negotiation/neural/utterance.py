from cocoa.neural.utterance import Utterance
from cocoa.neural.utterance import UtteranceBuilder as BaseUtteranceBuilder

from symbols import markers, category_markers
from core.price_tracker import PriceScaler
from cocoa.core.entity import is_entity

class UtteranceBuilder(BaseUtteranceBuilder):
    """
    Build a word-based utterance from the batch output
    of generator and the underlying dictionaries.
    """
    def build_target_tokens(self, predictions, kb=None):
        tokens = super(UtteranceBuilder, self).build_target_tokens(predictions, kb)
        tokens = [x for x in tokens if not x in category_markers]
        return tokens

    def _entity_to_str(self, entity_token, kb):
        raw_price = PriceScaler.unscale_price(kb, entity_token)
        human_readable_price = "${}".format(raw_price.canonical.value)
        return human_readable_price

    def get_price_number(self, entity, kb):
        raw_price = PriceScaler.unscale_price(kb, entity)
        return raw_price.canonical.value
