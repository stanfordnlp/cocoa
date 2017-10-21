from itertools import izip, izip_longest
import numpy as np

from cocoa.core.entity import is_entity
from cocoa.lib import logstats
from cocoa.model.util import EPS
from cocoa.model.evaluate import BaseEvaluator

from preprocess import markers
from graph import Graph

def remove_entities(entity_tokens):
    eoe_inds = [i for i, x in enumerate(entity_tokens) if x == markers.EOE]
    to_remove = set(eoe_inds)
    def find_entities(eoe_ind):
        i = eoe_ind - 1
        while i >= 0:
            if entity_tokens[i] != markers.EOS:
                to_remove.add(i)
            else:
                break
            i -= 1
    for eoe_ind in eoe_inds:
        find_entities(eoe_ind)
    return [x for i, x in enumerate(entity_tokens) if i not in to_remove], [x for i, x in enumerate(entity_tokens) if i in to_remove]

def pred_to_token(preds, stop_symbol, remove_symbols, textint_map, remove_entity, num_sents=None):
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
        if remove_entity:
            #print 'raw pred:', textint_map.int_to_text(pred, 'target')
            entity_tokens, prepended_entities = remove_entities(textint_map.int_to_text([x for x in pred[:find_stop(pred, n)]], 'target'))
            #tokens.append([x for x in entity_tokens if not x in (markers.EOS, markers.PAD)])
            tokens.append([x for x in entity_tokens if not x in (markers.PAD,)])
            entities.append(prepended_entities)
        else:
            tokens.append(textint_map.int_to_text([x for x in pred[:find_stop(pred, n)] if not x in remove_symbols], 'target'))
    return tokens, entities if len(entities) > 0 else None

class Evaluator(BaseEvaluator):
    def __init__(self, data, model, splits=('dev',), batch_size=1, verbose=True):
        super(Evaluator, self).__init__(data, model, splits, batch_size, verbose)
        self.copy = data.copy
        self.prepend = data.prepend

    def _stop_symbol(self):
        return self.vocab.to_ind(markers.EOS)

    def _remove_symbols(self):
        return map(self.vocab.to_ind, (markers.PAD,))

    def _generate_response(self, sess, dialogue_batch, summary_map):
        encoder_init_state = None
        # Whether we're using knowledge graphs
        graphs = dialogue_batch.get('graph', None)
        utterances = None
        for batch in dialogue_batch['batch_seq']:
            targets = batch['targets']
            max_len = targets.shape[1] + 10
            output_dict = self.model.generate(sess, batch, encoder_init_state, max_len, graphs=graphs, utterances=utterances, vocab=self.vocab, copy=self.copy, textint_map=self.data.textint_map)
            preds = output_dict['preds']
            true_final_state = output_dict['true_final_state']
            if graphs:
                encoder_init_state = true_final_state
                utterances = output_dict['utterances']
            else:
                encoder_init_state = true_final_state
            if self.copy:
                preds = graphs.copy_preds(preds, self.vocab.size)
            num_sents = np.sum(targets == self.stop_symbol, axis=1)
            pred_tokens, pred_entities = pred_to_token(preds, self.stop_symbol, self.remove_symbols, self.data.textint_map, self.prepend, num_sents)

            references = [self._process_target_tokens(tokens) for tokens in batch['decoder_tokens']]

            # Metrics
            # Sentence bleu: only for verbose print
            bleu_scores = self.sentence_bleu_score(pred_tokens, references)
            self.update_bleu_stats(summary_map, pred_tokens, references)
            self.update_entity_stats(summary_map, pred_tokens, references, 'entity_')
            if 'selection_scores' in output_dict:
                self.update_selection_stats(summary_map, output_dict['selection_scores'], output_dict['true_checklists'][:, -1, :], 'select_')
            if pred_entities is not None:
                self.update_entity_stats(summary_map, pred_entities, references, 'prepend_')

            if self.verbose:
                attn_scores = output_dict.get('attn_scores', None)
                probs = output_dict.get('probs', None)
                self._print_batch(batch, pred_tokens, references, bleu_scores, graphs, attn_scores, probs)

    def get_stats(self, summary_map):
        output = super(Evaluator, self).get_stats(summary_map)
        output['entity_f1'] = self.get_f1(summary_map, 'entity_')
        output['selection_f1'] = self.get_f1(summary_map, 'select_')
        output['prepend_f1'] = self.get_f1(summary_map, 'prepend_')
        return output

    def stats2str(self, stats):
        s = [super(Evaluator, self).stats2str(stats)]
        for m in ('entity_f1', 'selection_f1', 'prepend_f1'):
            s.append('%s=%.4f/%.4f/%.4f' % (m, stats[m][0], stats[m][1],stats[m][2]))
        return ' '.join(s)

    def update_selection_stats(self, summary_map, scores, targets, prefix=''):
        # NOTE: targets are from ground truth response and many contain new entities.
        # Ideally this would not happen as a mentioned entity is either from the agent's
        # KB or from partner's mentions (which is added to the graph), so during decoding
        # there shouldn't be new entities. However, the lexicon may "create" an entity.
        batch_size, num_nodes = scores.shape
        targets = targets[:, :num_nodes]

        pos_pred = scores > 0
        pos_target = targets == 1
        tp = np.sum(np.logical_and(pos_pred, pos_target))
        logstats.update_summary_map(summary_map, {prefix+'tp': tp, prefix+'pos_pred': np.sum(pos_pred), prefix+'pos_target': np.sum(pos_target)})

    def log_dict(self, stats):
        d = super(Evaluator, self).log_dict(stats)
        precision, recall, f1 = stats['entity_f1']
        d.update({'entity_precision': precision, 'entity_recall': recall, 'entity_f1': f1})
        return d

    def _process_target_tokens(self, tokens):
        targets = super(Evaluator, self)._process_target_tokens(tokens)
        if self.prepend:
            targets, _ = remove_entities(targets)
        return targets

    def _print_batch(self, batch, preds, targets, bleu_scores, graphs, attn_scores, probs):
        '''
        inputs are integers; targets and preds are tokens (converted in test_bleu).
        '''
        encoder_tokens = batch['encoder_tokens']
        inputs = batch['encoder_inputs']
        decoder_tokens = batch['decoder_tokens']
        print '-------------- batch ----------------'
        for i, (target, pred, bleu) in enumerate(izip_longest(targets, preds, bleu_scores)):
            # Skip padded turns
            if len(decoder_tokens[i]) == 0:
                continue
            print i
            if graphs:
                graphs.graphs[i].kb.dump()
            print 'RAW INPUT:', encoder_tokens[i]
            print 'RAW TARGET:', target
            print '----------'
            print 'INPUT:', self.data.textint_map.int_to_text(inputs[i], 'encoding')
            print 'TARGET:', target
            print 'PRED:', pred
            print 'BLEU:', bleu
            if probs is not None:
                print 'TOP-K:'
                for j, w in enumerate(pred):
                    print j
                    topk = np.argsort(probs[j][i])[::-1][:5]
                    for id_ in topk:
                        prob = probs[j][i][id_]
                        if id_ < self.vocab.size:
                            print self.vocab.to_word(id_), prob
                        else:
                            print graphs.graphs[i].nodes.to_word(id_ - self.vocab.size), prob
            #if attn_scores is not None:
            #    print 'ATTENTION:'
            #    for j, w in enumerate(pred):
            #        print 'TOKEN', j, w
            #        sorted_scores = sorted([(node_id, score) for node_id, score in enumerate(attn_scores[j][i])], key=lambda x: x[1], reverse=True)
            #        for node_id, score in sorted_scores:
            #            try:
            #                print node_id, graphs.graphs[i].nodes.to_word(node_id), score
            #            except KeyError:
            #                print node_id, 'pad', score


    # NOTE: both batch_preds and batch_targets must use canonical entity form: (name, type)
    def update_entity_stats(self, summary_map, batch_preds, batch_targets, prefix=''):
        def get_entity(x):
            return [e for e in x if is_entity(e)]
        pos_target = prefix + 'pos_target'
        pos_pred = prefix + 'pos_pred'
        tp = prefix + 'tp'
        for preds, targets in izip(batch_preds, batch_targets):
            preds = set(get_entity(preds))
            targets = set(get_entity(targets))
            # Don't record cases where no entity is presented
            if len(targets) > 0:
                logstats.update_summary_map(summary_map, {pos_target: len(targets), pos_pred: len(preds)})
                logstats.update_summary_map(summary_map, {tp: sum([1 if e in preds else 0 for e in targets])})

class FactEvaluator(object):
    '''
    Evaluate if a statement is true (approximately) given a KB.
    '''
    def __init__(self):
        keys = ('undecided', 'fact', 'single_fact', 'joint_fact', 'coref', 'correct_single', 'correct_joint', 'correct_joint_ent', 'repeated', 'same_col')
        self.summary_map = {}
        for k in keys:
            logstats.update_summary_map(self.summary_map, {k: 0})

    def inc_undecided(self):
        logstats.update_summary_map(self.summary_map, {'undecided': 1})

    def inc_fact(self):
        logstats.update_summary_map(self.summary_map, {'fact': 1})

    def inc_coref(self):
        logstats.update_summary_map(self.summary_map, {'coref': 1})

    def str_to_num(self, token):
        if token == 'no':
            return 0
        elif token == 'one':
            return 1
        elif token == 'two':
            return 2
        elif token == '3':
            return 3
        elif token == 'most':
            return 4
        elif token == 'all':
            return 5
        return None

    def eval_single(self, kb, span):
        #print 'eval_single:', span
        logstats.update_summary_map(self.summary_map, {'single_fact': 1})
        num, ent = span
        ent = ent[1]  # take the canonical form
        num = self.str_to_num(num)
        count = 0
        for i, item in enumerate(kb.items):
            for entity in self.item_entities(item):
                if entity == ent:
                    count += 1
        if num == count:
            #print 'correct single'
            logstats.update_summary_map(self.summary_map, {'correct_single': 1})

    def item_entities(self, item):
        attrs = sorted(item.items(), key=lambda x: x[0])
        for attr_name, value in attrs:
            type_ = Graph.metadata.attribute_types[attr_name]
            yield (value.lower(), type_)

    def eval_joint(self, kb, span):
        #print 'eval_joint:', span
        logstats.update_summary_map(self.summary_map, {'joint_fact': 1})
        num, ent1, _, ent2 = span
        ent1 = ent1[1]
        ent2 = ent2[1]
        if ent1 == ent2:
            #print 'repeated'
            logstats.update_summary_map(self.summary_map, {'repeated': 1})
            return
        # Same type, i.e. in the same column
        if ent1[1] == ent2[1]:
            #print 'same column'
            logstats.update_summary_map(self.summary_map, {'same_col': 1})
            return
        num = self.str_to_num(num)
        count = 0
        for i, item in enumerate(kb.items):
            entities = [entity for entity in self.item_entities(item)]
            if ent1 in entities and ent2 in entities:
                count += 1
        #print 'correct joint ent'
        logstats.update_summary_map(self.summary_map, {'correct_joint_ent': 1})
        if count == num:
            #print 'correct joint'
            logstats.update_summary_map(self.summary_map, {'correct_joint': 1})

    def report(self):
        num_total_facts = float(self.summary_map['fact']['sum']) + EPS
        num_single_facts = float(self.summary_map['single_fact']['sum']) + EPS
        num_joint_facts = float(self.summary_map['joint_fact']['sum']) + EPS
        result = {
                'undecided': self.summary_map['undecided']['sum'] / num_total_facts,
                'single_facts': self.summary_map['single_fact']['sum'] / num_total_facts,
                'joint_facts': self.summary_map['joint_fact']['sum'] / num_total_facts,
                'correct_single': self.summary_map['correct_single']['sum'] / num_single_facts,
                'correct_joint': self.summary_map['correct_joint']['sum'] / num_joint_facts,
                'correct_ent': self.summary_map['correct_joint_ent']['sum'] / num_joint_facts,
                'repeated': self.summary_map['repeated']['sum'] / num_joint_facts,
                'same_col': self.summary_map['same_col']['sum'] / num_joint_facts,
                'coref': self.summary_map['coref']['sum'] / num_total_facts,
                }
        return result

    def eval(self, kb, utterance):
        '''
        utterance: a list of tokens and entities represented as a tuple (surface_form, (caninical_form, type))
        '''
        #print 'eval:', utterance
        N = len(utterance)
        i = 0
        while i < N:
            token = utterance[i]
            if is_entity(token) and token[1][1] != 'item':
                self.inc_fact()
                if i+1 < N and utterance[i+1] == 'and':
                    # number ent1 and ent2
                    if i-1 < 0 or i+3 > N:
                        self.inc_undecided()
                        i += 1
                    else:
                        start, end = i-1, i+3
                        if not is_entity(utterance[i+2]):
                            self.inc_undecided()
                        else:
                            if end + 1 < N and utterance[end:end+2] == ['in', 'those']:
                                self.inc_coref()
                                i = end + 2
                            else:
                                self.eval_joint(kb, utterance[start:end])
                        i = end
                elif i-1 > 0:
                    # number ent
                    start, end = i-1, i+1
                    if end + 1 < N and utterance[end:end+2] == ['in', 'those']:
                        self.inc_coref()
                        i = end + 2
                    else:
                        self.eval_single(kb, utterance[start:end])
                        i = end
                else:
                    self.inc_undecided()
                    i += 1
            else:
                i += 1
