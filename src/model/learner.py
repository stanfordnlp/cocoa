'''
Main learning loop.
'''

import os
import time
import tensorflow as tf
from src.lib import logstats
import resource
import numpy as np
from model.util import EPS

def memory():
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return (usage[2]*resource.getpagesize()) / 1000000.0

def add_learner_arguments(parser):
    parser.add_argument('--optimizer', default='sgd', help='Optimization method')
    parser.add_argument('--unconditional', action='store_true', help='Do not pass final state to next batch')
    parser.add_argument('--sample-targets', action='store_true', help='Sample targets from candidates')
    parser.add_argument('--grad-clip', type=int, default=5, help='Min and max values of gradients')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--min-epochs', type=int, default=10, help='Number of training epochs to run before checking for early stop')
    parser.add_argument('--max-epochs', type=int, default=50, help='Maximum number of training epochs')
    parser.add_argument('--num-per-epoch', type=int, default=None, help='Number of examples per epoch')
    parser.add_argument('--print-every', type=int, default=1, help='Number of examples between printing training loss')
    parser.add_argument('--init-from', help='Initial parameters')
    parser.add_argument('--checkpoint', default='.', help='Directory to save learned models')
    parser.add_argument('--mappings', default='.', help='Directory to save mappings/vocab')
    parser.add_argument('--gpu', type=int, default=0, help='Use GPU or not')

optim = {'adagrad': tf.train.AdagradOptimizer,
         'sgd': tf.train.GradientDescentOptimizer,
         'adam': tf.train.AdamOptimizer,
        }

class BaseLearner(object):
    def __init__(self, data, model, evaluator, batch_size=1, unconditional=False, verbose=False):
        self.data = data  # DataGenerator object
        self.model = model
        self.vocab = data.mappings['vocab']
        self.batch_size = batch_size
        self.evaluator = evaluator
        self.unconditional = unconditional
        self.verbose = verbose

    def _run_batch(self, dialogue_batch, sess, summary_map, test=True):
        raise NotImplementedError

    def test_loss(self, sess, test_data, num_batches):
        '''
        Return the cross-entropy loss.
        '''
        summary_map = {}
        for i in xrange(num_batches):
            dialogue_batch = test_data.next()
            self._run_batch(dialogue_batch, sess, summary_map, test=True)
        return summary_map['total_loss']['sum'] / (summary_map['num_tokens']['sum'] + EPS)


    # TODO: factor this
    def _print_batch(self, batch, preds, loss):
        encoder_tokens = batch['encoder_tokens']
        encoder_inputs = batch['encoder_inputs']
        decoder_inputs = batch['decoder_inputs']
        decoder_tokens = batch['decoder_tokens']
        targets = batch['targets']
        # Go over each example in the batch
        print '-------------- batch ----------------'
        for i in xrange(encoder_inputs.shape[0]):
            if len(decoder_tokens[i]) == 0:
                continue
            print i
            print 'RAW INPUT:', encoder_tokens[i]
            print 'RAW TARGET:', decoder_tokens[i]
            print '----------'
            print 'ENC INPUT:', self.data.textint_map.int_to_text(encoder_inputs[i], 'encoding')
            print 'DEC INPUT:', self.data.textint_map.int_to_text(decoder_inputs[i], 'decoding')
            #print 'PRICE INPUT:', batch['decoder_price_inputs'][i]
            print 'TARGET:', self.data.textint_map.int_to_text(targets[i], 'target')
            #print 'PRICE TARGET:', batch['price_targets'][i]
            print 'PRED:', self.data.textint_map.int_to_text(preds[i], 'target')
            print 'LOSS:', loss[i]

    def eval(self, sess, name, test_data, num_batches):
        print '================== Eval %s ==================' % name
        results = {}

        # TODO: print_batch doesn't work for model=lm
        if (not name == 'test') and self.model.perplexity:
            print '================== Perplexity =================='
            start_time = time.time()
            loss = self.test_loss(sess, test_data, num_batches)
            results['loss'] = loss
            print 'loss=%.4f time(s)=%.4f' % (loss, time.time() - start_time)

        if name == 'test':
        #if True:
            print '================== Sampling =================='
            start_time = time.time()
            res = self.evaluator.test_response_generation(sess, test_data, num_batches)
            results.update(res)
            # TODO: hacky. for LM only.
            if len(results) > 0:
                print '%s time(s)=%.4f' % (self.evaluator.stats2str(results), time.time() - start_time)
        return results

    def learn(self, args, config, stats_file, ckpt=None, split='train'):
        logstats.init(stats_file)
        assert args.min_epochs <= args.max_epochs

        assert args.optimizer in optim.keys()
        optimizer = optim[args.optimizer](args.learning_rate)

        # Gradient
        grads_and_vars = optimizer.compute_gradients(self.model.loss)
        if args.grad_clip > 0:
            min_grad, max_grad = -1.*args.grad_clip, args.grad_clip
            clipped_grads_and_vars = [
                (tf.clip_by_value(grad, min_grad, max_grad) if grad is not None else grad, var) \
                for grad, var in grads_and_vars]
        else:
            clipped_grads_and_vars = grads_and_vars
        self.grad_norm = tf.global_norm([grad for grad, var in grads_and_vars])
        self.clipped_grad_norm = tf.global_norm([grad for grad, var in clipped_grads_and_vars])
        self.grad_norm = self.clipped_grad_norm

        # Optimize
        self.train_op = optimizer.apply_gradients(clipped_grads_and_vars)

        # Training loop
        train_data = self.data.generator(split, self.batch_size)
        num_per_epoch = train_data.next()
        step = 0
        saver = tf.train.Saver()
        save_path = os.path.join(args.checkpoint, 'tf_model.ckpt')
        best_saver = tf.train.Saver(max_to_keep=1)
        best_checkpoint = args.checkpoint+'-best'
        if not os.path.isdir(best_checkpoint):
            os.mkdir(best_checkpoint)
        best_save_path = os.path.join(best_checkpoint, 'tf_model.ckpt')
        best_loss = float('inf')
        # Number of iterations without any improvement
        num_epoch_no_impr = 0

        # Testing
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            if args.init_from:
                saver.restore(sess, ckpt.model_checkpoint_path)
            summary_map = {}
            #for epoch in xrange(args.max_epochs):
            epoch = 1
            while True:
                print '================== Epoch %d ==================' % (epoch)
                for i in xrange(num_per_epoch):
                    start_time = time.time()
                    self._run_batch(train_data.next(), sess, summary_map, test=False)
                    end_time = time.time()
                    logstats.update_summary_map(summary_map, \
                            {'time(s)/batch': end_time - start_time, \
                             'memory(MB)': memory()})
                    step += 1
                    if step % args.print_every == 0 or step % num_per_epoch == 0:
                        print '{}/{} (epoch {}) {}'.format(i+1, num_per_epoch, epoch, logstats.summary_map_to_str(summary_map))
                        summary_map = {}  # Reset
                step = 0

                # Save model after each epoch
                print 'Save model checkpoint to', save_path
                saver.save(sess, save_path, global_step=epoch)

                # Evaluate on dev
                for split, test_data, num_batches in self.evaluator.dataset():

                    results = self.eval(sess, split, test_data, num_batches)

                    # Start to record no improvement epochs
                    loss = results['loss']
                    if split == 'dev' and epoch > args.min_epochs:
                        if loss < best_loss * 0.995:
                            num_epoch_no_impr = 0
                        else:
                            num_epoch_no_impr += 1

                    if split == 'dev' and loss < best_loss:
                        print 'New best model'
                        best_loss = loss
                        best_saver.save(sess, best_save_path)
                        self.log_results('best_model', results)
                        logstats.add('best_model', {'epoch': epoch})
                        #logstats.add('best_model', {'bleu-4': bleu[0], 'bleu-3': bleu[1], 'bleu-2': bleu[2], 'entity_precision': ent_prec, 'entity_recall': ent_recall, 'entity_f1': ent_f1, 'loss': loss, 'epoch': epoch})

                # Early stop when no improvement
                if (epoch > args.min_epochs and num_epoch_no_impr >= 5) or epoch > args.max_epochs:
                    break
                epoch += 1

    def log_results(self, name, results):
        logstats.add(name, {'loss': results.get('loss', None)})
        logstats.add(name, self.evaluator.log_dict(results))



############# dynamic import depending on task ##################
import src.config as config
import importlib
task_module = importlib.import_module('.'.join(('src.model', config.task, 'learner')))
#Learner = task_module.Learner
get_learner = task_module.get_learner
