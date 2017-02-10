from itertools import izip, izip_longest
import numpy as np
from src.model.preprocess import markers
from src.model.graph import Graph
from src.lib.bleu import compute_bleu
from src.lib.bleu import bleu_stats as get_bleu_stats
from src.lib.bleu import bleu as get_bleu
from src.model.vocab import is_entity
from src.lib import logstats
from src.model.util import EPS

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
        tokens.append(textint_map.int_to_text([x for x in pred[:find_stop(pred, n)] if not x in remove_symbols], 'target'))
    return tokens, entities if len(entities) > 0 else None

class Evaluator(object):
    def __init__(self, data, model, splits=('dev',), batch_size=1, verbose=True):
        self.model = model
        self.batch_size = batch_size
        self.data = data
        self.vocab = data.mappings['vocab']
        self.verbose = verbose
        self.copy = data.copy

        # Prepare dataset
        self.eval_data = {split: data.generator(split, self.batch_size, shuffle=False) for split in splits}
        self.num_batches = {split: data.next() for split, data in self.eval_data.iteritems()}

        # For post-processing of generated utterances
        self.stop_symbol = self.vocab.to_ind(markers.EOS)
        #self.remove_symbols = map(self.vocab.to_ind, (markers.EOS, markers.PAD))
        self.remove_symbols = map(self.vocab.to_ind, (markers.PAD,))

    def dataset(self):
        '''
        Iterator over all datasets.
        '''
        for split, generator in self.eval_data.iteritems():
            yield split, generator, self.num_batches[split]

    def test_bleu(self, sess, test_data, num_batches):
        '''
        Go through each message of the agent and try to predict it
        given the *perfect* past.
        Return the average BLEU score across messages.
        '''
        summary_map = {}
        bleu_stats = [0 for i in xrange(10)]
        for i in xrange(num_batches):
            dialogue_batch = test_data.next()
            encoder_init_state = None
            # Whether we're using knowledge graphs
            if 'graph' in dialogue_batch:
                graphs = dialogue_batch['graph']
            else:
                graphs = None
            utterances = None
            for batch in dialogue_batch['batch_seq']:
                targets = batch['targets']
                max_len = targets.shape[1] + 10
                #preds, _, true_final_state, utterances, attn_scores = self.model.generate(sess, batch, encoder_init_state, max_len, graphs=graphs, utterances=utterances, vocab=self.vocab, copy=self.copy, textint_map=self.data.textint_map)
                output_dict = self.model.generate(sess, batch, encoder_init_state, max_len, graphs=graphs, utterances=utterances, vocab=self.vocab, copy=self.copy, textint_map=self.data.textint_map)
                preds = output_dict['preds']
                true_final_state = output_dict['true_final_state']
                if graphs:
                    encoder_init_state = true_final_state[0]
                    utterances = output_dict['utterances']
                else:
                    encoder_init_state = true_final_state
                if self.copy:
                    preds = graphs.copy_preds(preds, self.vocab.size)
                num_sents = np.sum(targets == self.stop_symbol, axis=1)
                pred_tokens, pred_entities = pred_to_token(preds, self.stop_symbol, self.remove_symbols, self.data.textint_map, num_sents)

                # Compute BLEU
                references = [self._process_target_tokens(tokens) for tokens in batch['decoder_tokens']]
                # Sentence bleu: only for verbose print
                bleu_scores = self.sentence_bleu_score(pred_tokens, references)
                bleu_stats = self.update_bleu_stats(bleu_stats, pred_tokens, references)
                self.update_entity_stats(summary_map, pred_tokens, references, 'entity_')

                if self.verbose:
                    attn_scores = output_dict.get('attn_scores', None)
                    probs = output_dict.get('probs', None)
                    self._print_batch(batch, pred_tokens, references, bleu_scores, graphs, attn_scores, probs)

        entity_f1 = self.get_f1(summary_map, 'entity_')
        bleu = (get_bleu(bleu_stats), get_bleu(bleu_stats[:-2]), get_bleu(bleu_stats[:-4]))
        return bleu, entity_f1

    def get_f1(self, summary_map, prefix):
        pos_target = prefix + 'pos_target'
        pos_pred = prefix + 'pos_pred'
        tp = prefix + 'tp'
        if tp not in summary_map:
            return -1, -1, -1
        tp, target_size, pred_size = float(summary_map[tp]['sum']), float(summary_map[pos_target]['sum']), float(summary_map[pos_pred]['sum'])
        return self._f1(tp, target_size, pred_size)

    def _f1(self, tp, pred_size, target_size):
        # This means no entity is detected in the test data. Probably something wrong.
        if target_size == 0 or pred_size == 0:
            return -1, -1 , -1
        recall = tp / target_size
        precision = tp / pred_size
        if recall + precision == 0:
            f1 = -1
        else:
            f1 = 2 * recall * precision / (recall + precision)
        return precision, recall, f1

    def _process_target_tokens(self, tokens):
        '''
        TODO: for now evaluate against canonical entities. In future, evaluate against
        actual utterances.
        '''
        targets = [token[1] if is_entity(token) else token for token in tokens]
        targets = [x for x in targets if x not in (markers.PAD,)]
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

    def update_summary(self, summary_map, bleu_scores):
        for bleu_score in bleu_scores:
            # None means no entity in this utterance
            if bleu_score is not None:
                logstats.update_summary_map(summary_map, {'bleu': bleu_score})

    def update_bleu_stats(self, stats, batch_preds, batch_targets):
        for preds, targets in izip(batch_preds, batch_targets):
            if len(targets) > 0:
                stats = [sum(scores) for scores in izip(stats, get_bleu_stats(preds, targets))]
        return stats

    def sentence_bleu_score(self, batch_preds, batch_targets):
        scores = []
        for preds, targets in izip(batch_preds, batch_targets):
            if len(targets) > 0:
                scores.append(compute_bleu(preds, targets))
            else:
                scores.append(None)
        return scores

    # NOTE: both batch_preds and batch_targets must use canonical entity form: (name, type)
    def update_entity_stats(self, summary_map, batch_preds, batch_targets, prefix=''):
        def get_entity(x):
            return [e for e in x if is_entity(e)]
        pos_target = prefix + 'pos_target'
        pos_pred = prefix + 'pos_pred'
        tp = prefix + 'tp'
        for preds, targets in izip (batch_preds, batch_targets):
            # None targets means that this is a padded turn
            if targets is None:
                recalls.append(None)
            else:
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
