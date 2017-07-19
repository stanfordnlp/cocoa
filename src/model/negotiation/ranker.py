from itertools import izip
import random
from src.lib.bleu import compute_bleu
import numpy as np
from preprocess import markers
from retriever import Retriever

def add_ranker_arguments(parser):
    parser.add_argument('--ranker', choices=['ir', 'cheat', 'encdec'], help='Ranking model')

class BaseRanker(object):
    def __init__(self):
        self.name = 'ranker'
        self.perplexity = False

    def select(self, batch):
        raise NotImplementedError

class IRRanker(BaseRanker):
    def __init__(self):
        super(IRRanker, self).__init__()
        self.name = 'ranker-ir'

    @classmethod
    def select(cls, batch):
        batch_candidates = batch['token_candidates']

        responses = [c[0].get('response', []) if len(c) > 0 else [] for c in batch_candidates]
        return {
                'responses': responses,
                'candidates': batch['token_candidates'],
                }
        #responses = []
        #for c in batch['token_candidates']:
        #    if len(c) == 0:
        #        responses.append([])
        #    else:
        #        r = np.random.choice([x for x in c if 'response' in x])
        #        responses.append(r['response'])
        #return responses

class CheatRanker(BaseRanker):
    def __init__(self):
        super(CheatRanker, self).__init__()
        self.name = 'ranker-cheat'

    @classmethod
    def select(cls, batch):
        candidates = batch['token_candidates']
        targets = batch['decoder_tokens']
        responses = []
        for c, target in izip(candidates, targets):
            if not len(target) > 0:
                response = {}
            else:
                scores = [compute_bleu(r.get('response', []), target) for r in c]
                if len(scores) == 0:
                    response = {}
                else:
                    response = c[np.argmax(scores)]
            responses.append(response.get('response', []))
        return responses

class EncDecRanker(BaseRanker):
    def __init__(self, model):
        super(EncDecRanker, self).__init__()
        self.model = model
        self.name = 'ranker-encdec'

    def set_tf_session(self, sess):
        self.sess = sess

    def _get_feed_dict_args(self, batch, encoder_init_state=None):
        encoder_args = {'inputs': batch['encoder_inputs'],
                'init_state': encoder_init_state,
                }
        decoder_args = {'inputs': batch['decoder_inputs'],
                'targets': batch['targets'],
                'context': batch['context'],
                }
        kwargs = {'encoder': encoder_args,
                'decoder': decoder_args,
                }
        return kwargs

    def score(self, candidates, kwargs, states=False):
        #candidates = batch['candidates']
        batch_size, num_candidate, _ = candidates.shape
        candidates_loss = np.zeros([batch_size, num_candidate])  # (batch_size, num_candidates)
        #if kwargs is None:
        #    kwargs = self._get_feed_dict_args(batch, encoder_init_state)
        final_states = []
        for i in xrange(num_candidate):
            candidate = candidates[:, i, :]  # (batch_size, seq_len)
            kwargs['decoder']['inputs'] = candidate[:, :-1]
            kwargs['decoder']['targets'] = candidate[:, 1:]
            feed_dict = self.model.get_feed_dict(**kwargs)
            if not states:
                batch_loss = self.sess.run(self.model.seq_loss, feed_dict=feed_dict)
            else:
                batch_loss, final_state = self.sess.run((self.model.seq_loss, self.model.final_state), feed_dict=feed_dict)
                final_states.append(final_state)
            candidates_loss[:, i] = batch_loss
        return candidates_loss, final_states

    def sample_candidates(self, candidates_loss):
        batch_size, num_candidate = candidates_loss.shape
        exp_x = np.exp(-1. * candidates_loss)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        best_candidates = []
        for i in xrange(batch_size):
            try:
                best_candidates.append(np.random.choice(num_candidate, 1, p=probs[i])[0])
            except ValueError:
                best_candidates.append(np.argmax(probs[i]))
        return best_candidates

    def select(self, batch, encoder_init_state):
        token_candidates = batch['token_candidates']
        kwargs = self._get_feed_dict_args(batch, encoder_init_state)

        candidates_loss, _ = self.score(batch['candidates'], kwargs)

        # Never choose empty ones
        for b, candidates in enumerate(token_candidates):
            for i, cand in enumerate(candidates):
                if 'response' not in cand:
                    candidates_loss[b][i] = 20.

        # Filter <accept>/<reject>
        # TODO: filter this in search
        prev_utterances = batch['encoder_tokens']
        offered = []
        for u in prev_utterances:
            if len(u) > 0 and markers.OFFER in u:
                offered.append(True)
            else:
                offered.append(False)
        for b, candidates in enumerate(token_candidates):
            has_offered = offered[b]
            for i, cand in enumerate(candidates):
                if 'response' in cand and (not has_offered) and (markers.ACCEPT in cand['response'] or markers.REJECT in cand['response']):
                    candidates_loss[b][i] = 20.

        #best_candidates = np.argmax(-1. * candidates_loss, axis=1)
        best_candidates = self.sample_candidates(candidates_loss)

        responses = [token_candidates[i][j]['response']
                if j < len(token_candidates[i]) and 'response' in token_candidates[i][j] else []
                for i, j in enumerate(best_candidates)]

        # Decoder the true utterance to get the state
        kwargs['decoder']['inputs'] = batch['decoder_inputs']
        kwargs['decoder']['targets'] = batch['targets']
        feed_dict = self.model.get_feed_dict(**kwargs)
        true_final_state = self.sess.run(self.model.final_state, feed_dict=feed_dict)

        return {
                'responses': responses,
                'true_final_state': true_final_state,
                'cheat_responses': CheatRanker.select(batch),
                'IR_responses': IRRanker.select(batch)['responses'][0],
                'candidates': token_candidates,
                }

