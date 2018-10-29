from itertools import izip
import random
from cocoa.lib.bleu import compute_bleu
import numpy as np
from preprocess import markers
from retriever import Retriever

def add_ranker_arguments(parser):
    parser.add_argument('--ranker', choices=['ir', 'cheat', 'encdec', 'sf'], help='Ranking model')
    parser.add_argument('--temperature', default=0., type=float, help='Temperature for sampling candidates')

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

        responses = [c[0] if len(c) > 0 else [] for c in batch_candidates]
        return responses
        return {
                'responses': responses,
                'candidates': batch['token_candidates'],
                }

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
    def __init__(self, model, temp):
        super(EncDecRanker, self).__init__()
        self.model = model
        self.temp = temp
        self.name = 'ranker-encdec'

    def set_tf_session(self, sess):
        self.sess = sess

    def _get_feed_dict_args(self, batch, encoder_init_state=None):
        encoder_args = {'inputs': batch['encoder_inputs'],
                'init_state': encoder_init_state,
                'context': batch['encoder_context'],
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
        if self.temp == 0:
            return np.argmax(-1. * candidates_loss, axis=1)

        batch_size, num_candidate = candidates_loss.shape
        exp_x = np.exp(-1. * candidates_loss / self.temp)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        best_candidates = []
        for i in xrange(batch_size):
            try:
                best_candidates.append(np.random.choice(num_candidate, 1, p=probs[i])[0])
            except ValueError:
                best_candidates.append(np.argmax(probs[i]))
        return best_candidates

    def select(self, batch, encoder_init_state, textint_map=None):
        token_candidates = batch['token_candidates']
        kwargs = self._get_feed_dict_args(batch, encoder_init_state)

        candidates_loss, _ = self.score(batch['candidates'], kwargs)

        # Never choose empty ones
        for b, candidates in enumerate(token_candidates):
            for i, cand in enumerate(candidates):
                if 'response' not in cand:
                    candidates_loss[b][i] = 20.

        best_candidates = self.sample_candidates(candidates_loss)

        responses = [token_candidates[i][j]['response']
                if j < len(token_candidates[i]) and 'response' in token_candidates[i][j] else []
                for i, j in enumerate(best_candidates)]

        # Decoder the true utterance to get the state
        if self.model.stateful:
            kwargs['decoder']['inputs'] = batch['decoder_inputs']
            kwargs['decoder']['targets'] = batch['targets']
            feed_dict = self.model.get_feed_dict(**kwargs)
            true_final_state = self.sess.run(self.model.final_state, feed_dict=feed_dict)
        else:
            true_final_state = None

        return {
                'responses': responses,
                'true_final_state': true_final_state,
                'cheat_responses': CheatRanker.select(batch),
                'IR_responses': IRRanker.select(batch)['responses'][0],
                'candidates': token_candidates,
                }

class SlotFillingRanker(EncDecRanker):
    def __init__(self, model):
        super(SlotFillingRanker, self).__init__(model)
        self.name = 'ranker-sf'

    def rewrite(self, batch, encoder_init_state, textint_map):
        candidates = batch['candidates']
        batch_size, num_candidate, _ = candidates.shape

        # Encoding
        encoder_args = {'inputs': batch['encoder_inputs'],
                'init_cell_state': encoder_init_state,
                }
        encoder_output_dict = self.model.encoder.run_encode(self.sess, **encoder_args)

        decoder_args = self.model.decoder.get_inference_args(batch, encoder_output_dict, textint_map)

        true_inputs = decoder_args['inputs']

        rewritten_candidates = []
        for i in xrange(num_candidate):
            candidate = candidates[:, i, :]  # (batch_size, seq_len)
            decoder_args['inputs'] = candidate
            decoder_output_dict = self.model.decoder.run_decode(self.sess, **decoder_args)
            rewritten_candidates.append(decoder_output_dict['preds'][0])

        # True final state
        decoder_args['inputs'] = true_inputs
        feed_dict = self.model.decoder.get_feed_dict(**decoder_args)
        true_final_state = self.sess.run(self.model.final_state, feed_dict=feed_dict)

        return rewritten_candidates, true_final_state

    def select(self, batch, encoder_init_state, textint_map=None):
        rewritten_candidates, true_final_state = self.rewrite(batch, encoder_init_state, textint_map)
        token_rewritten_candidates = [textint_map.int_to_text(c) for c in rewritten_candidates]
        token_rewritten_candidates  = [' '.join([str(x) for x in c if x != markers.PAD])
                for c in token_rewritten_candidates]
        return {
                'responses': token_candidates[:1],
                'true_final_state': true_final_state,
                'candidates': batch['token_candidates'],
                'rewritten': token_rewritten_candidates,
                }
