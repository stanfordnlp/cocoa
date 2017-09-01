import numpy as np
from cocoa.lib import logstats
from cocoa.model.learner import BaseLearner

class Learner(BaseLearner):
    def __init__(self, data, model, evaluator, batch_size=1, verbose=False):
        super(Learner, self).__init__(data, model, evaluator, batch_size, verbose)
        if type(self.model).__name__ == 'BasicEncoderDecoder':
            setattr(self, '_run_batch', self._run_batch_basic)
        elif type(self.model).__name__ == 'GraphEncoderDecoder':
            setattr(self, '_run_batch', self._run_batch_graph)

    # TODO: don't need graphs in the parameters
    # TODO: refactor this, put in models
    def _get_feed_dict(self, batch, encoder_init_state=None, graph_data=None, graphs=None, copy=False, init_checklists=None, encoder_nodes=None, decoder_nodes=None, matched_items=None):
        # NOTE: We need to do the processing here instead of in preprocess because the
        # graph is dynamic; also the original batch data should not be modified.
        if copy:
            targets = graphs.copy_targets(batch['targets'], self.vocab.size)
            # NOTE: we're not using matched_items for now in the model
            matched_items = graphs.copy_targets(np.reshape(matched_items, [-1, 1]), self.vocab.size)
            matched_items = np.reshape(matched_items, [-1])
        else:
            targets = batch['targets']

        encoder_args = {'inputs': batch['encoder_inputs'],
                'last_inds': batch['encoder_inputs_last_inds'],
                'init_state': encoder_init_state,
                }
        decoder_args = {'inputs': batch['decoder_inputs'],
                'last_inds': batch['decoder_inputs_last_inds'],
                'targets': targets,
                }
        kwargs = {'encoder': encoder_args,
                'decoder': decoder_args,
                }

        if graph_data is not None:
            encoder_args['update_entities'] = graph_data['encoder_entities']
            decoder_args['update_entities'] = graph_data['decoder_entities']
            encoder_args['utterances'] = graph_data['utterances']
            kwargs['graph_embedder'] = graph_data
            decoder_args['init_checklists'] = init_checklists
            encoder_args['entities'] = encoder_nodes
            decoder_args['entities'] = decoder_nodes
            decoder_args['cheat_selection'] = decoder_nodes
            decoder_args['encoder_entities'] = encoder_nodes

        feed_dict = self.model.get_feed_dict(**kwargs)
        return feed_dict

    def _run_batch_graph(self, dialogue_batch, sess, summary_map, test=False):
        '''
        Run truncated RNN through a sequence of batch examples with knowledge graphs.
        '''
        encoder_init_state = None
        utterances = None
        graphs = dialogue_batch['graph']
        matched_items = dialogue_batch['matched_items']
        for i, batch in enumerate(dialogue_batch['batch_seq']):
            graph_data = graphs.get_batch_data(batch['encoder_tokens'], batch['decoder_tokens'], batch['encoder_entities'], batch['decoder_entities'], utterances, self.vocab)
            init_checklists = graphs.get_zero_checklists(1)
            feed_dict = self._get_feed_dict(batch, encoder_init_state, graph_data, graphs, self.data.copy, init_checklists, graph_data['encoder_nodes'], graph_data['decoder_nodes'], matched_items)
            if test:
                logits, final_state, utterances, loss, seq_loss, total_loss, sel_loss = sess.run(
                        [self.model.decoder.output_dict['logits'],
                         self.model.final_state,
                         self.model.decoder.output_dict['utterances'],
                         self.model.loss, self.model.seq_loss, self.model.total_loss, self.model.select_loss],
                        feed_dict=feed_dict)
            else:
                _, logits, final_state, utterances, loss, seq_loss, sel_loss, gn = sess.run(
                        [self.train_op,
                         self.model.decoder.output_dict['logits'],
                         self.model.final_state,
                         self.model.decoder.output_dict['utterances'],
                         self.model.loss,
                         self.model.seq_loss,
                         self.model.select_loss,
                         self.grad_norm], feed_dict=feed_dict)
            encoder_init_state = final_state

            if self.verbose:
                preds = np.argmax(logits, axis=2)
                if self.data.copy:
                    preds = graphs.copy_preds(preds, self.data.mappings['vocab'].size)
                self._print_batch(batch, preds, seq_loss)

            if test:
                logstats.update_summary_map(summary_map, {'total_loss': total_loss[0], 'num_tokens': total_loss[1]})
            else:
                logstats.update_summary_map(summary_map, {'loss': loss})
                logstats.update_summary_map(summary_map, {'sel_loss': sel_loss})
                logstats.update_summary_map(summary_map, {'grad_norm': gn})

    def _run_batch_basic(self, dialogue_batch, sess, summary_map, test=False):
        '''
        Run truncated RNN through a sequence of batch examples.
        '''
        encoder_init_state = None
        matched_items = dialogue_batch['matched_items']
        for batch in dialogue_batch['batch_seq']:
            feed_dict = self._get_feed_dict(batch, encoder_init_state, matched_items=matched_items)
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
