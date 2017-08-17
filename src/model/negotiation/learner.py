import numpy as np
from src.lib import logstats
from src.model.learner import BaseLearner
from src.model.negotiation.ranker import EncDecRanker
from src.lib.bleu import compute_bleu

def get_learner(data_generator, model, evaluator, batch_size=1, verbose=False, sample_targets=False):
    if sample_targets:
        return PseudoTargetLearner(data_generator, model, evaluator, batch_size=batch_size, verbose=verbose)
    elif model.name == 'lm':
        return LMLearner(data_generator, model, evaluator, batch_size=batch_size, verbose=verbose)
    #if model.name == 'ranker':
    #    return RetrievalLearner(data_generator, model, evaluator, batch_size=batch_size, verbose=verbose)
    else:
        return Learner(data_generator, model, evaluator, batch_size=batch_size, verbose=verbose)

class Learner(BaseLearner):
    def __init__(self, data, model, evaluator, batch_size=1, verbose=False):
        super(Learner, self).__init__(data, model, evaluator, batch_size, verbose)

    def _get_feed_dict(self, batch, encoder_init_state=None, init_price_history=None, test=False):
        # NOTE: We need to do the processing here instead of in preprocess because the
        # graph is dynamic; also the original batch data should not be modified.
        targets = batch['targets']

        price_args = {'init_price_history': init_price_history,
                }
        encoder_args = {'inputs': batch['encoder_inputs'],
                'context': batch['encoder_context'],
                'init_state': encoder_init_state,
                'price_inputs': batch['encoder_price_inputs'],
                'price_predictor': price_args,
                }
        # NOTE: decoder get init_price_history from the encoder output, so no input here
        decoder_args = {'inputs': batch['decoder_inputs'],
                'targets': targets,
                'price_inputs': batch['decoder_price_inputs'],
                'price_targets': batch['price_targets'],
                'price_predictor': {},
                'context': batch['context'],
                }
        kwargs = {'encoder': encoder_args,
                'decoder': decoder_args,
                }

        feed_dict = self.model.get_feed_dict(**kwargs)

        if test:
            feed_dict[self.model.keep_prob] = 1.0

        return feed_dict

    def _run_batch(self, dialogue_batch, sess, summary_map, test=False):
        '''
        Run truncated RNN through a sequence of batch examples.
        '''
        encoder_init_state = None
        init_price_history = None
        for batch in dialogue_batch['batch_seq']:
            # TODO: hacky
            if init_price_history is None and hasattr(self.model.decoder, 'price_predictor'):
                batch_size = batch['encoder_inputs'].shape[0]
                init_price_history = self.model.decoder.price_predictor.zero_init_price(batch_size)
            feed_dict = self._get_feed_dict(batch, encoder_init_state, test=test, init_price_history=init_price_history)
            fetches = {
                    'logits': self.model.decoder.output_dict['logits'],
                    'loss': self.model.loss,
                    'seq_loss': self.model.seq_loss,
                    }
            if not test:
                fetches['train_op'] = self.train_op
                fetches['gn'] = self.grad_norm
            else:
                fetches['total_loss'] = self.model.total_loss
            if self.model.stateful:
                fetches['final_state'] = self.model.final_state
            if hasattr(self.model.decoder, 'price_predictor'):
                fetches['price_history'] = self.model.decoder.output_dict['price_history']

            results = sess.run(fetches, feed_dict=feed_dict)

            if self.model.stateful:
                encoder_init_state = results['final_state']
            else:
                encoder_init_state = None
            if 'price_history' in results:
                init_price_history = results['price_history']

            if self.verbose:
                preds = np.argmax(results['logits'], axis=2)
                self._print_batch(batch, preds, results['seq_loss'])

            if test:
                total_loss = results['total_loss']
                logstats.update_summary_map(summary_map, {'total_loss': total_loss[0], 'num_tokens': total_loss[1]})
            else:
                logstats.update_summary_map(summary_map, {'loss': results['loss']})
                logstats.update_summary_map(summary_map, {'grad_norm': results['gn']})

class PseudoTargetLearner(Learner):
    def __init__(self, data, model, evaluator, batch_size=1, verbose=False):
        super(PseudoTargetLearner, self).__init__(data, model, evaluator, batch_size, verbose)
        self.ranker = EncDecRanker(model)

    def _get_feed_dict(self, batch, encoder_init_state=None, init_price_history=None):
        # Sample a target
        if not self.test:
            candidates = batch['candidates']
            kwargs = self.ranker._get_feed_dict_args(batch, encoder_init_state)
            candidates_loss, _ = self.ranker.score(candidates, kwargs)  # (batch_size, num_candidates)
            best_candidates = self.ranker.sample_candidates(candidates_loss)
            #for i, c in enumerate(best_candidates):
            #    print 'TARGET:'
            #    print batch['decoder_tokens'][i]
            #    print 'CANDIDATE TARGET:'
            #    print batch['token_candidates'][i][c]['response']

            batch_size, _, seq_len = candidates.shape
            new_targets = np.zeros([batch_size, seq_len])
            for i, c in enumerate(best_candidates):
                new_targets[i] = candidates[i, c]

            new_batch = dict(batch)
            new_batch['decoder_inputs'] = new_targets[:, :-1]
            new_batch['targets'] = new_targets[:, 1:]
        else:
            new_batch = batch

        return super(PseudoTargetLearner, self)._get_feed_dict(new_batch, encoder_init_state, init_price_history)

    def _run_batch(self, dialogue_batch, sess, summary_map, test=False):
        '''
        Run truncated RNN through a sequence of batch examples.
        '''
        self.ranker.set_tf_session(sess)
        self.test = test
        super(PseudoTargetLearner, self)._run_batch(dialogue_batch, sess, summary_map, test)

class LMLearner(Learner):
    def _get_feed_dict(self, batch, init_state=None, init_price_history=None):
        kwargs = {
                'inputs': batch['inputs'],
                'targets': batch['targets'],
                'init_state': init_state,
                'context': batch['context'],
                }
        feed_dict = self.model.get_feed_dict(**kwargs)
        return feed_dict
