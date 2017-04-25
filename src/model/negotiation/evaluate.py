import numpy as np
from itertools import izip
from preprocess import markers
from src.model.evaluate import BaseEvaluator

# TODO: factor this
def pred_to_token(preds, stop_symbol, remove_symbols, textint_map, num_sents=None):
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
    for pred, n in izip(preds, num_sents):
        tokens.append(textint_map.int_to_text([x for x in pred[:find_stop(pred, n)] if not x in remove_symbols]))
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
            true_final_state = output_dict['true_final_state']
            encoder_init_state = true_final_state
            num_sents = np.sum(targets == self.stop_symbol, axis=1)
            pred_tokens, pred_entities = pred_to_token(preds, self.stop_symbol, self.remove_symbols, self.data.textint_map, num_sents)

            references = [self._process_target_tokens(tokens) for tokens in batch['decoder_tokens']]

            # Metrics
            # Sentence bleu: only for verbose print
            bleu_scores = self.sentence_bleu_score(pred_tokens, references)
            self.update_bleu_stats(summary_map, pred_tokens, references)
            #self.update_entity_stats(summary_map, pred_tokens, references, 'entity_')

            if self.verbose:
                #attn_scores = output_dict.get('attn_scores', None)
                probs = output_dict.get('probs', None)
                # TODO: print
                #self._print_batch(batch, pred_tokens, references, bleu_scores, graphs, attn_scores, probs)

