'''
Main learning loop.
'''

import os
import time
import resource
import tensorflow as tf
import numpy as np

from cocoa.lib import logstats
from cocoa.model.util import EPS

def memory():
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return (usage[2]*resource.getpagesize()) / 1000000.0

def add_learner_arguments(parser):
    parser.add_argument('--optimizer', default='sgd', help='Optimization method')
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
    parser.add_argument('--summary-dir', default='/tmp', help='Path to summary logs')
    parser.add_argument('--eval-modes', nargs='*', default=('loss',), help='What to evaluate {loss, generation}')

optim = {'adagrad': tf.train.AdagradOptimizer,
         'sgd': tf.train.GradientDescentOptimizer,
         'adam': tf.train.AdamOptimizer,
        }

class Learner(object):
    def __init__(self, data, model, evaluator, batch_size=1, summary_dir='/tmp', verbose=False):
        self.data = data  # DataGenerator object
        self.model = model
        self.vocab = data.mappings['vocab']
        self.batch_size = batch_size
        self.evaluator = evaluator
        self.verbose = verbose
        self.summary_dir = summary_dir

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
        return summary_map

    def collect_summary_test(self, summary_map, results={}):
        results['loss'] = summary_map['total_loss']['sum'] / (summary_map['num_tokens']['sum'] + EPS)
        return results

    def collect_summary_train(self, summary_map, results={}):
        # Mean of batch statistics
        results.update({k: summary_map[k]['mean'] for k in ('loss', 'grad_norm')})
        return results

    def _print_batch(self, batch, preds, loss):
        batcher = self.data.dialogue_batcher
        textint_map = self.data.textint_map
        # Go over each example in the batch
        print '-------------- Batch ----------------'
        for i in xrange(batch['size']):
            success = batcher.print_batch(batch, i, textint_map, preds)
        print 'BATCH LOSS:', loss

    def eval(self, sess, name, test_data, num_batches, output=None, modes=('loss',)):
        print '================== Eval %s ==================' % name
        results = {}

        if 'loss' in modes:
            print '================== Loss =================='
            start_time = time.time()
            summary_map = self.test_loss(sess, test_data, num_batches)
            results = self.collect_summary_test(summary_map, results)
            results_str = ' '.join(['{}={:.4f}'.format(k, v) for k, v in results.iteritems()])
            print '%s time(s)=%.4f' % (results_str, time.time() - start_time)

        if 'generation' in modes:
            print '================== Generation =================='
            start_time = time.time()
            res = self.evaluator.test_response_generation(sess, test_data, num_batches, output=output)
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
        self.global_step = 0

        # Testing
        with tf.Session(config=config) as sess:
            # Summary
            self.merged_summary = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(self.summary_dir, sess.graph)

            sess.run(tf.global_variables_initializer())
            if args.init_from:
                saver.restore(sess, ckpt.model_checkpoint_path)
            summary_map = {}
            epoch = 1
            while True:
                print '================== Epoch %d ==================' % (epoch)
                for i in xrange(num_per_epoch):
                    start_time = time.time()
                    self._run_batch(train_data.next(), sess, summary_map, test=False)
                    end_time = time.time()
                    results = self.collect_summary_train(summary_map)
                    results['time(s)/batch'] = end_time - start_time
                    results['memory(MB)'] = memory()
                    results_str = ' '.join(['{}={:.4f}'.format(k, v) for k, v in sorted(results.items())])
                    step += 1
                    if step % args.print_every == 0 or step % num_per_epoch == 0:
                        print '{}/{} (epoch {}) {}'.format(i+1, num_per_epoch, epoch, results_str)
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

                # Early stop when no improvement
                if (epoch > args.min_epochs and num_epoch_no_impr >= 5) or epoch > args.max_epochs:
                    break
                epoch += 1

    def log_results(self, name, results):
        logstats.add(name, {'loss': results.get('loss', None)})
        logstats.add(name, self.evaluator.log_dict(results))
