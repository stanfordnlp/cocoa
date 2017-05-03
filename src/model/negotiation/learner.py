import numpy as np
from src.lib import logstats
from src.model.learner import BaseLearner

class Learner(BaseLearner):
    def __init__(self, data, model, evaluator, batch_size=1, verbose=False):
        super(Learner, self).__init__(data, model, evaluator, batch_size, verbose)

    def _get_feed_dict(self, batch, encoder_init_state=None):
        # NOTE: We need to do the processing here instead of in preprocess because the
        # graph is dynamic; also the original batch data should not be modified.
        targets = batch['targets']

        encoder_args = {'inputs': batch['encoder_inputs'],
                'init_state': encoder_init_state,
                }
        decoder_args = {'inputs': batch['decoder_inputs'],
                'targets': targets,
                }
        kwargs = {'encoder': encoder_args,
                'decoder': decoder_args,
                }

        feed_dict = self.model.get_feed_dict(**kwargs)
        return feed_dict

    def _run_batch(self, dialogue_batch, sess, summary_map, test=False):
        '''
        Run truncated RNN through a sequence of batch examples.
        '''
        encoder_init_state = None
        for batch in dialogue_batch['batch_seq']:
            feed_dict = self._get_feed_dict(batch, encoder_init_state)
            if test:
                logits, final_state, loss, seq_loss, total_loss = sess.run([
                    self.model.decoder.output_dict['logits'],
                    self.model.final_state,
                    self.model.loss, self.model.seq_loss, self.model.total_loss],
                    feed_dict=feed_dict)
            else:
                _, logits, final_state, loss, seq_loss, gn = sess.run([
                    self.train_op,
                    self.model.decoder.output_dict['logits'],
                    self.model.final_state,
                    self.model.loss, self.model.seq_loss,
                    self.grad_norm], feed_dict=feed_dict)
            encoder_init_state = final_state

            if self.verbose:
                preds = np.argmax(logits, axis=2)
                self._print_batch(batch, preds, seq_loss)

            if test:
                logstats.update_summary_map(summary_map, {'total_loss': total_loss[0], 'num_tokens': total_loss[1]})
            else:
                logstats.update_summary_map(summary_map, {'loss': loss})
                logstats.update_summary_map(summary_map, {'grad_norm': gn})
