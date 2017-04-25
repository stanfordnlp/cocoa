from itertools import izip
from src.lib.bleu import compute_bleu
from src.lib.bleu import bleu_stats as get_bleu_stats
from src.lib.bleu import bleu as get_bleu

class BaseEvaluator(object):
    def __init__(self, data, model, splits=('dev',), batch_size=1, verbose=True):
        self.model = model
        self.batch_size = batch_size
        self.data = data
        self.vocab = data.mappings['vocab']
        self.verbose = verbose

        # Prepare dataset
        self.eval_data = {split: data.generator(split, self.batch_size, shuffle=False) for split in splits}
        self.num_batches = {split: data.next() for split, data in self.eval_data.iteritems()}

        self.stop_symbol = self._stop_symbol()
        self.remove_symbols = self._remove_symbols()

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

    def sentence_bleu_score(self, batch_preds, batch_targets):
        scores = []
        for preds, targets in izip(batch_preds, batch_targets):
            if len(targets) > 0:
                scores.append(compute_bleu(preds, targets))
            else:
                scores.append(None)
        return scores

    def _generate_response(sess, dialogue_batch, summary_map):
        raise NotImplementedError

    def get_stats(self, summary_map):
        bleu_stats = summary_map['bleu_stats']
        bleu = (get_bleu(bleu_stats), get_bleu(bleu_stats[:-2]), get_bleu(bleu_stats[:-4]))
        return {'bleu': bleu}

    def stats2str(self, stats):
        bleu = stats['bleu']
        return 'bleu=%.4f/%.4f/%.4f' % (bleu[0], bleu[1], bleu[2])

    def log_dict(self, stats):
        '''
        Return a dict of metrics to be logged.
        '''
        return {'bleu-4': stats['bleu'][0], 'bleu-3': stats['bleu'][1], 'bleu-2': stats['bleu'][2]}

    def test_response_generation(self, sess, test_data, num_batches):
        '''
        Go through each message of the agent and try to predict it given the *perfect* past.
        '''
        summary_map = {}
        bleu_stats = [0 for i in xrange(10)]
        summary_map['bleu_stats'] = bleu_stats

        for i in xrange(num_batches):
            dialogue_batch = test_data.next()
            self._generate_response(sess, dialogue_batch, summary_map)

        return self.get_stats(summary_map)

import src.config as config
import importlib
task_module = importlib.import_module('.'.join(('src.model', config.task, 'evaluate')))
Evaluator = task_module.Evaluator
pred_to_token = task_module.pred_to_token
