import numpy as np
from itertools import izip, izip_longest, takewhile

from cocoa.lib import logstats
from cocoa.core.entity import is_entity
from cocoa.model.evaluate import BaseEvaluator
from cocoa.model.util import safe_div

from core.price_tracker import PriceTracker
from preprocess import markers, Dialogue, category_to_marker

def get_evaluator(data_generator, model, splits=('test',), batch_size=1, verbose=True):
    if model.name in ('ranker-cheat', 'ranker-ir'):
        return RetrievalEvaluator(data_generator, model, splits, batch_size, verbose)
    elif model.name in ('ranker-encdec',):
        return EncDecRetrievalEvaluator(data_generator, model, splits, batch_size, verbose)
    elif model.name == 'lm':
        return LMEvaluator(data_generator, model, splits, batch_size, verbose)
    elif model.name == 'selector':
        return SelectorEvaluator(data_generator, model, splits, batch_size, verbose)
    elif model.name == 'ir':
        return IREvaluator(data_generator, model, splits, batch_size, verbose)
    else:
        return EncDecEvaluator(data_generator, model, splits, batch_size, verbose)


class Evaluator(BaseEvaluator):
    """Basic evaluator: common functions for this task.
    """
    @classmethod
    def pred_to_token(cls, preds, stop_symbol, remove_symbols, textint_map, num_sents=None, prices=None):
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

    def _stop_symbol(self):
        return self.vocab.to_ind(markers.EOS)

    def _remove_symbols(self):
        inds = map(self.vocab.to_ind, (markers.PAD,))
        words = [markers.PAD]
        return inds + words

    def add_batch_results(self, batch, batch_id, prev_turns, references, preds):
        """Add evaluation result for each example in the batch.

        Args:
            batch (dict): a batch from DialogueBatcher
            batch_id (int): index of the batch in batch_seq
            prev_turns (list[list]): list of batch utterances at each time step (seq_len, batch_size)
            references (list[list]): reference utterances (batch_size, ...)
            preds (list[list]): predicted utterances (batch_size, ...)

        """
        for i in xrange(batch['size']):
            target = batch['decoder_tokens'][i]
            # Padded turn
            if not target:
                continue
            uuid = batch['uuids'][i]
            role = batch['kbs'][i].facts['personal']['Role']
            example_id = '{uuid}-{role}-{turn}'.format(uuid=uuid, role=role, turn=batch_id)
            context = [turns[i] for turns in prev_turns]
            self.add_results(example_id=example_id, prev_turns=context, target=references[i], pred=preds[i])

    def generate_response(self, dialogue_batch, summary_map, model_vars, eval_callback):
        """Generate response for each example in the batch.

        Args:
            dialoge_batch (dict): batch from DialogueBatcher
            summary_map (dict): update metrics (see logstats)
            model_vars (dict): variables specific to the model being evaluated
            eval_callback (func): callback function that takes the generation output and optionally updates model_vars in-between baches

        """
        prev_turns = []
        textint_map = self.data.textint_map
        for i, batch in enumerate(dialogue_batch['batch_seq']):
            prev_turns.append(batch['encoder_tokens'])

            output_dict = self._generate_response(batch, model_vars, textint_map)
            eval_callback(output_dict, model_vars)

            pred_tokens = [self._process_target_tokens(tokens) for tokens in output_dict['pred_tokens']]
            references = [self._process_target_tokens(tokens) for tokens in batch['decoder_tokens']]

            self.add_batch_results(batch, i, prev_turns, references, pred_tokens)

            # Metrics
            # Sentence bleu: only for verbose print
            bleu_scores = self.sentence_bleu_score(pred_tokens, references)
            self.update_bleu_stats(summary_map, pred_tokens, references)
            self.update_entity_stats(summary_map, pred_tokens, references, 'entity_')

            if self.verbose:
                best_candidates = output_dict.get('candidate_ranks', None)
                self._print_batch(batch, pred_tokens, references, bleu_scores, best_candidates=best_candidates)

            prev_turns.append(batch['decoder_tokens'])

    def get_stats(self, summary_map):
        output = super(Evaluator, self).get_stats(summary_map)
        output['entity_f1'] = self.get_f1(summary_map, 'entity_')
        #if 'multi_score' in summary_map:
        #    output['multi_score'] = summary_map['multi_score']['mean']
        output['multi_score'] = self.multi_bleu_score(self.preds, self.references)[0]
        output['single_score'] = self.multi_bleu_score(self.preds, self.targets)[0]
        output['most_similar_score'] = self.multi_bleu_score(self.preds, self.most_similar_references)[0]
        return output

    def stats2str(self, stats):
        s = [super(Evaluator, self).stats2str(stats)]
        for m in ('entity_f1',):
            s.append('%s=%.4f/%.4f/%.4f' % (m, stats[m][0], stats[m][1],stats[m][2]))
        if 'multi_score' in stats:
            s.append('%s=%.4f' % ('multi_score', stats['multi_score']))
            s.append('%s=%.4f' % ('single_score', stats['single_score']))
            s.append('%s=%.4f' % ('most_similar_score', stats['most_similar_score']))
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
        if 'entity_f1' in stats:
            precision, recall, f1 = stats['entity_f1']
            d.update({'entity_precision': precision, 'entity_recall': recall, 'entity_f1': f1})
        return d

    def _process_target_tokens(self, tokens):
        remove_tokens = [markers.GO_B, markers.GO_S] + category_to_marker.values()
        process_entity = lambda e: e.canonical if e.canonical.type == 'price' else e.surface
        targets = [process_entity(token) if is_entity(token) else token for token in tokens if not token in remove_tokens]
        return targets

    def to_str(self, words):
        return ' '.join([str(w) for w in words if w not in self.remove_symbols])

    def _print_batch(self, batch, preds, targets, bleu_scores, best_candidates=None):
        """Print the generation results.

        Args:
            batch (dict): from DialogueBatcher
            preds (list[list[str]]): predicted utterances
            targets (list[list[str]]): reference utterences

        """
        batcher = self.data.dialogue_batcher
        textint_map = self.data.textint_map
        print '-------------- Batch ----------------'
        for i, (pred, bleu) in enumerate(izip_longest(preds, bleu_scores)):
            success = batcher.print_batch(batch, i, textint_map, best_candidates)
            if success:
                print 'PRED:\n {}'.format(batcher.list_to_text(pred))
                print 'BLEU:', bleu


class EncDecEvaluator(Evaluator):
    def __init__(self, data, model, splits=('dev',), batch_size=1, verbose=True):
        super(EncDecEvaluator, self).__init__(data, model, splits=splits, batch_size=batch_size, verbose=verbose)
        self.sess = None

    def multi_ref_scores(self, batch_candidates, batch_candidate_scores, batch_preds, batch_targets, summary_map):
        best_candidates = []
        for candidates, scores, preds, target in izip(batch_candidates, batch_candidate_scores, batch_preds, batch_targets):
            candidates = [c['response'] for c in candidates if 'response' in c]
            assert len(candidates) == len(scores)
            scores = [sum(s) for s in scores]
            candidates = [self._process_target_tokens(c) for i, c in enumerate(candidates) if scores[i] > 0]
            candidates.append(target)

            self.preds.append(preds)
            self.references.append(candidates)
            self.targets.append([target])

            bleus = [self.sentence_bleu_score([preds], [c])[0] for c in candidates]
            most_similar_candidate = np.argmax(bleus)
            best_candidates.append(candidates[most_similar_candidate])
            self.most_similar_references.append([candidates[most_similar_candidate]])
            #weighted_bleu = bleus[most_similar_candidate] * float(sum(scores[most_similar_candidate]))
            #print bleus[most_similar_candidate]
            #if bleus[most_similar_candidate] < 0.5:
            #    print 'preds:', preds
            #    print 'cand:', candidates[most_similar_candidate]
            #    #for c in candidates:
            #    #    print c
            #weighted_bleu = self.sentence_multi_bleu_score(preds, candidates)

            #logstats.update_summary_map(summary_map, {'multi_score': weighted_bleu})

        return best_candidates

    def generate_response(self, sess, dialogue_batch, summary_map):
        model_vars = {
                'sess': sess,
                'encoder_init_state': None,
                }
        def eval_callback(output_dict, model_vars):
            if self.model.stateful:
                model_vars['encoder_init_state'] = output_dict['true_final_state']
        return super(EncDecEvaluator, self).generate_response(dialogue_batch, summary_map, model_vars, eval_callback)

    def _generate_response(self, batch, model_vars, textint_map):
        max_len = 100
        sess = model_vars['sess']
        encoder_init_state = model_vars['encoder_init_state']

        output_dict = self.model.generate(sess, batch, encoder_init_state, max_len, textint_map=textint_map)
        preds = output_dict['preds']
        prices = output_dict['prices']

        num_sents = np.sum(batch['decoder_args']['targets'] == self.stop_symbol, axis=1)
        pred_tokens, pred_entities = self.pred_to_token(preds, self.stop_symbol, self.remove_symbols, self.data.textint_map, num_sents=num_sents, prices=prices)

        return {'pred_tokens': pred_tokens}


class SelectorEvaluator(Evaluator):
    """Evaluate candidate selector (retrieval-based models).
    """
    @classmethod
    def recall_at_k(cls, labels, scores, k=1, summary_map=None):
        """Recall of the top-k candidates.

        Args:
            labels: binary (batch_size, num_candidates), 1 means good candidate
            scores: ranking scores (batch_size, num_candidates)
            summary_map (bool): if true, accumulates relevant statistics.

        Returns:
            The percentage of good candidates in the top-k candidates.

        """
        topk_candidates = np.argsort(-1.*scores, axis=1)[:, :k]
        # num_candidates might be smaller than k
        batch_size, actual_k = topk_candidates.shape

        binary_preds = np.zeros_like(labels, dtype=np.int32)
        row_inds = np.tile(np.arange(batch_size), [actual_k, 1]).T
        binary_preds[row_inds, topk_candidates] = 1

        num_true_positive = np.sum(np.logical_and(labels == 1, labels == binary_preds))
        num_positive_example = np.sum(labels)
        recall = safe_div(float(num_true_positive), num_positive_example)

        if summary_map is not None:
            prefix = 'recall_at_{}'.format(k)
            logstats.update_summary_map(summary_map, {
                '{}_tp'.format(prefix): num_true_positive,
                '{}_p'.format(prefix): num_positive_example,
                })

        return recall

    @classmethod
    def recall_at_k_from_summary(self, k, summary_map):
        prefix = 'recall_at_{}'.format(k)
        num_true_positive = summary_map['{}_tp'.format(prefix)]['sum']
        num_positive_example = summary_map['{}_p'.format(prefix)]['sum']
        recall = safe_div(float(num_true_positive), num_positive_example)
        return recall

    def _generate_response(self, batch, model_vars, textint_map):
        sess = model_vars['sess']
        output_dict = self.model.generate(sess, batch, None)
        return {
                'candidate_ranks': output_dict['candidate_ranks'],
                'pred_tokens': output_dict['responses'],
                }

    def generate_response(self, sess, dialogue_batch, summary_map):
        model_vars = {
                'sess': sess,
                }
        def eval_callback(output_dict, model_vars):
            return
        return super(SelectorEvaluator, self).generate_response(dialogue_batch, summary_map, model_vars, eval_callback)

class IREvaluator(SelectorEvaluator):
    """Evaluator for the IR system.
    """
    def _generate_response(self, batch, model_vars, textint_map):
        candidate_ranks, responses = self.model.generate(batch)
        return {
                'candidate_ranks': candidate_ranks,
                'pred_tokens': responses,
                }

    def generate_response(self, sess, dialogue_batch, summary_map):
        model_vars = {}
        def eval_callback(output_dict, model_vars):
            return
        return super(SelectorEvaluator, self).generate_response(dialogue_batch, summary_map, model_vars, eval_callback)


class RetrievalEvaluator(Evaluator):
    def _generate_response(self, sess, dialogue_batch, summary_map):
        prev_turns = []
        for batch in dialogue_batch['batch_seq']:
            references = [self._process_target_tokens(tokens) for tokens in batch['decoder_tokens']]
            prev_turns.append([self._process_target_tokens(tokens) for tokens in batch['encoder_tokens']])
            output_dict = self.model.select(batch)
            pred_tokens = self._process_target_tokens(output_dict['responses'])

            if 'token_candidates' in batch:
                self.multi_ref_scores(batch['token_candidates'], batch['candidate_scores'], pred_tokens, summary_map)

            # Metrics
            # Sentence bleu: only for verbose print
            bleu_scores = self.sentence_bleu_score(pred_tokens, references)
            self.update_bleu_stats(summary_map, pred_tokens, references)
            self.update_entity_stats(summary_map, pred_tokens, references, 'entity_')

            if self.verbose:
                self._print_batch(batch, prev_turns, pred_tokens, references, bleu_scores, output_dict)
            prev_turns.append(references)

    def _print_batch(self, batch, prev_turns, preds, targets, bleu_scores, output_dict=None, results=None):
        '''
        inputs are integers; targets and preds are tokens (converted in test_bleu).
        '''
        encoder_tokens = batch['encoder_tokens']
        inputs = batch['encoder_inputs']
        decoder_tokens = batch['decoder_tokens']
        kbs = batch['kbs']

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
            print 'CONTEXT:'
            for turn in prev_turns[-3:]:
                print self.to_str(turn[i])
            #print 'INPUT:', self.to_str(self.data.textint_map.int_to_text(inputs[i], 'encoding'))
            #print 'TARGET:', Dialogue.original_price(kb, target)
            #print 'PRED:', Dialogue.original_price(kb, pred)
            print 'TARGET:', self.to_str(target)
            print 'PRED:', self.to_str(pred)
            print 'BLEU:', bleu
            print 'ALL CANDIDATES:'
            for c in output_dict['candidates'][i]:
                if c != {}:
                    print 'Hits:', c['hits']
                    print 'Response:', self.to_str(c['response'])

class EncDecRetrievalEvaluator(RetrievalEvaluator):
    def _generate_response(self, sess, dialogue_batch, summary_map):
        encoder_init_state = None
        prev_turns  =[]
        for batch in dialogue_batch['batch_seq']:
            references = [self._process_target_tokens(tokens) for tokens in batch['decoder_tokens']]
            prev_turns.append([self._process_target_tokens(tokens) for tokens in batch['encoder_tokens']])
            # TODO
            output_dict = self.model.select(batch, encoder_init_state, self.data.textint_map)
            pred_tokens = output_dict['responses']
            pred_tokens = [self._process_target_tokens(tokens) for tokens in pred_tokens]
            encoder_init_state = output_dict['true_final_state']
            references = [self._process_target_tokens(tokens) for tokens in batch['decoder_tokens']]

            # TODO: refactor
            if 'token_candidates' in batch:
                self.multi_ref_scores(batch['token_candidates'], batch['candidate_scores'], pred_tokens, references, summary_map)

            # Metrics
            # Sentence bleu: only for verbose print
            bleu_scores = self.sentence_bleu_score(pred_tokens, references)
            self.update_bleu_stats(summary_map, pred_tokens, references)
            self.update_entity_stats(summary_map, pred_tokens, references, 'entity_')

            if self.verbose:
                self._print_batch(batch, prev_turns, pred_tokens, references, bleu_scores, output_dict)
            prev_turns.append(references)

    def _print_batch(self, batch, prev_turns, preds, targets, bleu_scores, output_dict=None, results=None):
        '''
        inputs are integers; targets and preds are tokens (converted in test_bleu).
        '''
        encoder_tokens = batch['encoder_tokens']
        inputs = batch['encoder_inputs']
        decoder_tokens = batch['decoder_tokens']
        kbs = batch['kbs']

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
            print 'CONTEXT:'
            for turn in prev_turns[-3:]:
                print self.to_str(turn[i])
            #print 'INPUT:', self.to_str(self.data.textint_map.int_to_text(inputs[i], 'encoding'))
            #print 'TARGET:', Dialogue.original_price(kb, target)
            #print 'PRED:', Dialogue.original_price(kb, pred)
            print 'TARGET:', self.to_str(target)
            print 'PRED:', self.to_str(pred)
            #print 'BLEU:', bleu
            #print 'CHEAT:', self.to_str(output_dict['cheat_responses'][i])
            #print 'IR:', self.to_str(output_dict['IR_responses'][i])
            #print 'ALL CANDIDATES:'
            #for c in output_dict['candidates'][i]:
            #    if c != {}:
            #        #print 'Hits:', c['hits']
            #        print 'Response:', self.to_str(c['response'])


class LMEvaluator(Evaluator):
    def _stop_symbol(self):
        return self.vocab.to_ind(markers.EOS)

    def _remove_symbols(self):
        inds = map(self.vocab.to_ind, (markers.PAD,))
        words = [markers.PAD]
        return inds + words

    def _generate_response(self, sess, dialogue_batch, summary_map):
        init_state = None
        for batch in dialogue_batch['eval_batch_seq']:
            targets = batch['targets']
            max_len = targets.shape[1] + 10
            output_dict = self.model.generate(sess, batch, init_state, max_len, textint_map=self.data.textint_map)
            preds = output_dict['preds']
            true_final_state = output_dict['true_final_state']
            init_state = true_final_state
            num_sents = np.sum(targets == self.stop_symbol, axis=1)
            pred_tokens, pred_entities = self.pred_to_token(preds, self.stop_symbol, self.remove_symbols, self.data.textint_map, num_sents=num_sents)

            references = [self._process_target_tokens(tokens) for tokens in batch['decoder_tokens']]

            # Metrics
            # Sentence bleu: only for verbose print
            bleu_scores = self.sentence_bleu_score(pred_tokens, references)
            self.update_bleu_stats(summary_map, pred_tokens, references)
            self.update_entity_stats(summary_map, pred_tokens, references, 'entity_')

            if self.verbose:
                #attn_scores = output_dict.get('attn_scores', None)
                #probs = output_dict.get('probs', None)
                self._print_batch(batch, pred_tokens, references, bleu_scores)

    #def _print_batch(self, preds, targets, bleu_scores):
    #    for i, (target, pred, bleu) in enumerate(izip_longest(targets, preds, bleu_scores)):
    #        print '----------'
    #        #print 'INPUT:', self.to_str(self.data.textint_map.int_to_text(inputs[i], 'encoding'))
    #        print 'TARGET:', self.to_str(target)
    #        print 'PRED:', self.to_str(pred)
    #        print 'BLEU:', bleu
