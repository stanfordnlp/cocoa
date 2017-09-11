from itertools import izip

from cocoa.core.util import write_json
from cocoa.core.entity import is_entity
from cocoa.lib.bleu import compute_bleu
from cocoa.lib.bleu import bleu_stats as get_bleu_stats
from cocoa.lib.bleu import bleu as get_bleu
from cocoa.core.util import write_json
from cocoa.lib import multi_bleu

class BaseEvaluator(object):
    def __init__(self, data, model, splits=('dev',), batch_size=1, verbose=True):
        self.model = model
        self.batch_size = batch_size
        self.data = data
        self.vocab = data.mappings['vocab']
        self.verbose = verbose

        # Prepare dataset
        self.eval_data = {split: data.generator(split, shuffle=False) for split in splits}
        self.num_batches = {split: data.next() for split, data in self.eval_data.iteritems()}

        self.stop_symbol = self._stop_symbol()
        self.remove_symbols = self._remove_symbols()

        # TODO: hacky
        self.preds = []
        self.references = []
        self.targets = []
        self.most_similar_references = []

        self.eval_results = None

    def dump_results(self, output):
        """Write eval_results to a JSON file.
        """
        print 'Writing {} evaluation results to {}'.format(len(self.eval_results), output)
        write_json(self.eval_results, output)

    def add_results(self, example_id=None, prev_turns=None, target=None, pred=None, **kwargs):
        """Save response generation results.
        """
        if self.eval_results is None:
            self.eval_results = []
        result = {
                'ex_id': example_id,
                'prev_turns': prev_turns,
                'reference': target,
                'response': pred,
                }
        self.eval_results.append(result)

    def _stop_symbol(self):
        '''
        For post-processing of generated utterances: remove tokens after stop_symbol
        '''
        raise NotImplementedError

    def _remove_symbols(self):
        '''
        Remove these symbols in a sequence, e.g. UNK, PAD
        '''
        raise NotImplementedError

    def dataset(self):
        '''
        Iterator over all datasets.
        '''
        for split, generator in self.eval_data.iteritems():
            yield split, generator, self.num_batches[split]

    def update_bleu_stats(self, summary_map, batch_preds, batch_targets):
        stats = summary_map['bleu_stats']
        for preds, targets in izip(batch_preds, batch_targets):
            if len(targets) > 0:
                stats = [sum(scores) for scores in izip(stats, get_bleu_stats(preds, targets))]
        summary_map['bleu_stats'] = stats

    def multi_bleu_score(self, preds, targets):
        return multi_bleu.multi_bleu(preds, targets)

    def sentence_bleu_score(self, batch_preds, batch_targets):
        scores = []
        for preds, targets in izip(batch_preds, batch_targets):
            if len(targets) > 0:
                scores.append(compute_bleu(preds, targets))
            else:
                scores.append(None)
        return scores

    def generate_response(sess, dialogue_batch, summary_map):
        raise NotImplementedError

    def get_stats(self, summary_map):
        bleu_stats = summary_map['bleu_stats']
        bleu = (get_bleu(bleu_stats), get_bleu(bleu_stats[:-2]), get_bleu(bleu_stats[:-4]))
        return {'bleu': bleu}

    def stats2str(self, stats):
        if 'bleu' in stats:
            bleu = stats['bleu']
            return 'bleu=%.4f/%.4f/%.4f' % (bleu[0], bleu[1], bleu[2])
        else:
            return ''

    def log_dict(self, stats):
        '''
        Return a dict of metrics to be logged.
        '''
        if 'bleu' in stats:
            return {'bleu-4': stats['bleu'][0], 'bleu-3': stats['bleu'][1], 'bleu-2': stats['bleu'][2]}
        else:
            return {}

    def test_response_generation(self, sess, test_data, num_batches, output=None):
        '''
        Go through each message of the agent and try to predict it given the *perfect* past.
        '''
        summary_map = {}
        bleu_stats = [0 for i in xrange(10)]
        summary_map['bleu_stats'] = bleu_stats
        #results = []

        for i in xrange(num_batches):
            dialogue_batch = test_data.next()
            self.generate_response(sess, dialogue_batch, summary_map)

        if output:
            self.dump_results(output)

        return self.get_stats(summary_map)

    def _process_target_tokens(self, tokens):
        '''
        TODO: for now evaluate against canonical entities. In future, evaluate against
        actual utterances.
        '''
        targets = [token[1] if is_entity(token) else token for token in tokens]
        #targets = [x for x in targets if x not in (markers.PAD,)]
        return targets

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
