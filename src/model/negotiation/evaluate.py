import numpy as np
from itertools import izip, izip_longest
from src.lib import logstats
from preprocess import markers
from src.basic.entity import is_entity
from src.model.evaluate import BaseEvaluator
from preprocess import Dialogue, price_filler
from src.basic.negotiation.price_tracker import PriceTracker

# TODO: factor this
def pred_to_token(preds, stop_symbol, remove_symbols, textint_map, num_sents=None, prices=None):
    '''
    Convert integer predition to tokens. Remove PAD and EOS.
    preds: (batch_size, max_len)
    '''
    def find_stop(array, n):
        count = 0
        for i, a in enumerate(array):
            if a == stop_symbol:
                count += 1
                if count == n:
                    # +1: include </s>
                    return i + 1
        return None
    tokens = []
    entities = []
    if num_sents is None:
        num_sents = [1 for _ in preds]
    if prices:
        for pred, n, price in izip(preds, num_sents, prices):
            N = find_stop(pred, n)
            assert len(pred) == len(price)
            token_price = [(x, p) for x, p in izip(pred[:N], price[:N]) if not x in remove_symbols]
            s = textint_map.int_to_text([x[0] for x in token_price], prices=[x[1] for x in token_price])
    else:
        for pred, n in izip(preds, num_sents):
            s = textint_map.int_to_text([x for x in pred[:find_stop(pred, n)] if not x in remove_symbols])
            tokens.append(s)
    return tokens, entities if len(entities) > 0 else None

class Evaluator(BaseEvaluator):
    def _stop_symbol(self):
        return self.vocab.to_ind(markers.EOS)

    def _remove_symbols(self):
        return map(self.vocab.to_ind, (markers.PAD,))

    def _generate_response(self, sess, dialogue_batch, summary_map):
        encoder_init_state = None
        for batch in dialogue_batch['batch_seq']:
            targets = batch['targets']
            max_len = targets.shape[1] + 10
            output_dict = self.model.generate(sess, batch, encoder_init_state, max_len, textint_map=self.data.textint_map)
            preds = output_dict['preds']
            prices = output_dict['prices']
            true_final_state = output_dict['true_final_state']
            encoder_init_state = true_final_state
            num_sents = np.sum(targets == self.stop_symbol, axis=1)
            pred_tokens, pred_entities = pred_to_token(preds, self.stop_symbol, self.remove_symbols, self.data.textint_map, num_sents=num_sents, prices=prices)

            references = [self._process_target_tokens(tokens) for tokens in batch['decoder_tokens']]

            # Metrics
            # Sentence bleu: only for verbose print
            bleu_scores = self.sentence_bleu_score(pred_tokens, references)
            self.update_bleu_stats(summary_map, pred_tokens, references)
            self.update_entity_stats(summary_map, pred_tokens, references, 'entity_')

            if self.verbose:
                #attn_scores = output_dict.get('attn_scores', None)
                #probs = output_dict.get('probs', None)
                # TODO: print
                self._print_batch(batch, pred_tokens, references, bleu_scores)

    def _process_target_tokens(self, tokens):
        # TODO: hack
        #targets = ['<%s>' % token[1][1] if is_entity(token) else token for token in tokens]
        targets = [token[1] if is_entity(token) else token for token in tokens]
        return targets

    def _print_batch(self, batch, preds, targets, bleu_scores):
        '''
        inputs are integers; targets and preds are tokens (converted in test_bleu).
        '''
        encoder_tokens = batch['encoder_tokens']
        inputs = batch['encoder_inputs']
        decoder_tokens = batch['decoder_tokens']
        kbs = batch['kbs']
        def to_str(l):
            words = []
            for w in l:
                if is_entity(w):
                    words.append('[%s]' % PriceTracker.get_price(w))
                elif not w.startswith('<'):
                    words.append(w)
            return ' '.join(words)

        print '-------------- batch ----------------'
        for i, (target, pred, bleu) in enumerate(izip_longest(targets, preds, bleu_scores)):
            # Skip padded turns
            if len(decoder_tokens[i]) == 0:
                continue
            kb = kbs[i]
            kb.dump()
            #print 'RAW INPUT:', Dialogue.original_price(kb, encoder_tokens[i])
            #print 'RAW TARGET:', Dialogue.original_price(kb, target)
            #print 'RAW INPUT:', encoder_tokens[i]
            #print 'RAW TARGET:', target
            print '----------'
            print 'INPUT:', to_str(self.data.textint_map.int_to_text(inputs[i], 'encoding'))
            #print 'TARGET:', Dialogue.original_price(kb, target)
            #print 'PRED:', Dialogue.original_price(kb, pred)
            print 'TARGET:', to_str(target)
            print 'PRED:', to_str(pred)
            print 'BLEU:', bleu

    def get_stats(self, summary_map):
        output = super(Evaluator, self).get_stats(summary_map)
        output['entity_f1'] = self.get_f1(summary_map, 'entity_')
        return output

    def stats2str(self, stats):
        s = [super(Evaluator, self).stats2str(stats)]
        for m in ('entity_f1',):
            s.append('%s=%.4f/%.4f/%.4f' % (m, stats[m][0], stats[m][1],stats[m][2]))
        return ' '.join(s)

    # NOTE: both batch_preds and batch_targets must use canonical entity form: (name, type)
    def update_entity_stats(self, summary_map, batch_preds, batch_targets, prefix=''):
        def get_entity(x):
            return [e for e in x if is_entity(e)]
        pos_target = prefix + 'pos_target'
        pos_pred = prefix + 'pos_pred'
        tp = prefix + 'tp'
        for preds, targets in izip (batch_preds, batch_targets):
            preds = set(get_entity(preds))
            targets = set(get_entity(targets))
            # Don't record cases where no entity is presented
            if len(targets) > 0:
                logstats.update_summary_map(summary_map, {pos_target: len(targets), pos_pred: len(preds)})
                logstats.update_summary_map(summary_map, {tp: sum([1 if e in preds else 0 for e in targets])})

    def log_dict(self, stats):
        d = super(Evaluator, self).log_dict(stats)
        precision, recall, f1 = stats['entity_f1']
        d.update({'entity_precision': precision, 'entity_recall': recall, 'entity_f1': f1})
        return d

class CheatRetrievalEvaluator(Evaluator):
    def __init__(self, data, model, responses, splits=('dev',), batch_size=1, verbose=True):
        super(CheatRetrievalEvaluator, self).__init__(data, model, splits, batch_size, verbose)
        self.responses = responses

    def retrieve(self, cheat_target, role):
        responses = self.responses[role]
        cheat_targets = [cheat_target] * len(responses)
        scores = self.sentence_bleu_score(responses, cheat_targets)
        return responses[np.argmax(scores)]

    def _generate_response(self, sess, dialogue_batch, summary_map):
        kbs = dialogue_batch['kb']
        roles = [kb.facts['personal']['Role'] for kb in kbs]
        for batch in dialogue_batch['batch_seq']:
            references = [self._process_target_tokens(tokens) for tokens in batch['decoder_tokens']]
            pred_tokens = [self.retrieve(target, role) for target, role in izip(references, roles)]

            # Metrics
            # Sentence bleu: only for verbose print
            bleu_scores = self.sentence_bleu_score(pred_tokens, references)
            self.update_bleu_stats(summary_map, pred_tokens, references)
            self.update_entity_stats(summary_map, pred_tokens, references, 'entity_')

            if self.verbose:
                #attn_scores = output_dict.get('attn_scores', None)
                #probs = output_dict.get('probs', None)
                # TODO: print
                self._print_batch(batch, pred_tokens, references, bleu_scores)

