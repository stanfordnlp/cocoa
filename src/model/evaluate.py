from itertools import izip
from src.model.preprocess import markers
from src.model.graph import Graph
from src.lib.bleu import compute_bleu
from src.lib.bleu import bleu_stats as get_bleu_stats
from src.lib.bleu import bleu as get_bleu
from src.model.vocab import is_entity
from src.lib import logstats

def pred_to_token(preds, stop_symbol, remove_symbols, textint_map):
    '''
    Convert integer predition to tokens. Remove PAD and EOS.
    preds: (batch_size, max_len)
    '''
    def find_stop(array):
        for i, a in enumerate(array):
            if a == stop_symbol:
                return i
        return None
    tokens = []
    for pred in preds:
        tokens.append(textint_map.int_to_text([x for x in pred[:find_stop(pred)] if not x in remove_symbols], 'target'))
    return tokens

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
        self.remove_symbols = map(self.vocab.to_ind, (markers.EOS, markers.PAD))

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
                max_len = batch['targets'].shape[1] + 10
                preds, _, true_final_state, utterances, attn_scores = self.model.generate(sess, batch, encoder_init_state, max_len, graphs=graphs, utterances=utterances, vocab=self.vocab, copy=self.copy, textint_map=self.data.textint_map)
                if graphs:
                    encoder_init_state = true_final_state[0]
                else:
                    encoder_init_state = true_final_state
                if self.copy:
                    preds = graphs.copy_preds(preds, self.vocab.size)
                pred_tokens = pred_to_token(preds, self.stop_symbol, self.remove_symbols, self.data.textint_map)

                # Compute BLEU
                references = [self._process_target_tokens(targets) for targets in batch['decoder_tokens']]
                # Sentence bleu: only for verbose print
                bleu_scores = self.sentence_bleu_score(pred_tokens, references)
                bleu_stats = self.update_bleu_stats(bleu_stats, pred_tokens, references)
                self.update_entity_stats(summary_map, pred_tokens, batch['decoder_tokens'])

                if self.verbose:
                    self._print_batch(batch, pred_tokens, references, bleu_scores, graphs, attn_scores)

        precision, recall, f1 = self.entity_f1(summary_map)
        bleu = get_bleu(bleu_stats)
        return bleu, precision, recall, f1

    def entity_f1(self, summary_map):
        tp, target_size, pred_size = float(summary_map['tp']['sum']), float(summary_map['target_size']['sum']), float(summary_map['pred_size']['sum'])
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
        return [token[1] if is_entity(token) else token for token in tokens]

    def _print_batch(self, batch, preds, targets, bleu_scores, graphs, attn_scores):
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
            print 'ATTENTION:'
            for j, w in enumerate(pred):
                print 'TOKEN', j, w
                sorted_scores = sorted([(node_id, score) for node_id, score in enumerate(attn_scores[j][i])], key=lambda x: x[1], reverse=True)
                for node_id, score in sorted_scores:
                    try:
                        print node_id, graphs.graphs[i].nodes.to_word(node_id), score
                    except KeyError:
                        print node_id, 'pad', score

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

    def update_entity_stats(self, summary_map, batch_preds, batch_targets):
        def get_entity(x):
            return [e[0] for e in x if is_entity(e)]
        for preds, targets in izip (batch_preds, batch_targets):
            # None targets means that this is a padded turn
            if targets is None:
                recalls.append(None)
            else:
                preds = set(get_entity(preds))
                targets = set(get_entity(targets))
                # Don't record cases where no entity is presented
                if len(targets) > 0:
                    logstats.update_summary_map(summary_map, {'target_size': len(targets), 'pred_size': len(preds)})
                    logstats.update_summary_map(summary_map, {'tp': sum([1 if e in preds else 0 for e in targets])})

