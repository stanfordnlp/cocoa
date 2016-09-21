'''
Main learning loop.
'''

import os, sys
import time
import tensorflow as tf
from preprocess import END_TURN, END_UTTERANCE
from lib.bleu import compute_bleu
from lib import logstats
import numpy as np
from vocab import is_entity

def add_learner_arguments(parser):
    parser.add_argument('--optimizer', default='sgd', help='Optimization method')
    parser.add_argument('--grad-clip', type=int, default=5, help='Min and max values of gradients')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--max-epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--num-per-epoch', type=int, help='Number of examples per epoch')
    parser.add_argument('--print-every', type=int, default=1, help='Number of examples between printing training loss')
    parser.add_argument('--init-from', help='Initial parameters')
    parser.add_argument('--checkpoint', default='.', help='Directory to save learned models')
    parser.add_argument('--gpu', type=int, default=0, help='Use GPU or not')

optim = {'adagrad': tf.train.AdagradOptimizer,
         'sgd': tf.train.GradientDescentOptimizer,
         'adam': tf.train.AdamOptimizer,
        }

class Learner(object):
    def __init__(self, data, model, verbose=False):
        self.data = data  # DataGenerator object
        self.model = model
        self.verbose = verbose

    def entity_acc(self, preds, targets, lexicon, vocab):
        def get_entity(x):
            x = map(vocab.to_word, x)
            return [e[0] for e in x if is_entity(e)]
        preds = set(get_entity(preds))
        targets = set(get_entity(targets))
        # Don't record cases where no entity is presented
        if len(targets) == 0:
            return None
        recall = sum([1 if e in preds else 0 for e in targets]) / float(len(targets))
        return recall

    def test_bleu(self, sess, split='dev'):
        '''
        Go through each message of the agent and try to predict it
        given the perfect past.
        Return the average BLEU score across messages.
        '''
        test_data = self.data.generator_eval(split)
        stop_symbols = map(self.data.vocab.to_ind, (END_TURN, END_UTTERANCE))
        #stop_symbols = []
        max_len = 20
        summary_map = {}
        for ex in test_data:
            agent, kb, inputs, entities, targets = ex
            preds, _ = self.model.generate(sess, kb, inputs, entities, stop_symbols, max_len)
            bleu = compute_bleu(preds, targets)
            ent_acc = self.entity_acc(preds, targets, self.data.lexicon, self.data.vocab)
            # ent_acc is None means targets has not entity
            if ent_acc is not None:
                logstats.update_summary_map(summary_map, {'acc': ent_acc})
            logstats.update_summary_map(summary_map, {'bleu': bleu})

            if self.verbose:
                #kb.dump()
                print 'AGENT=%d' % agent
                print 'INPUT:', map(self.data.vocab.to_word, list(inputs[0]))
                print 'TARGET:', map(self.data.vocab.to_word, targets)
                print 'PRED:', map(self.data.vocab.to_word, preds)
                print 'BLEU=%.2f' % bleu
        return summary_map['bleu']['mean'], summary_map['acc']['mean']

    def test_loss(self, sess, split='dev'):
        '''
        Return the cross-entropy loss.
        '''
        test_data = self.data.generator_train(split)
        summary_map = {}
        for i in xrange(2*self.data.num_examples[split]):
            feed_dict = self._get_feed_dict(test_data)
            output, loss = sess.run([self.model.outputs, self.model.loss], feed_dict=feed_dict)
            if self.verbose:
                pred = self.model.get_prediction(output)
                print 'PRED:', map(self.data.vocab.to_word, list(pred[0]))
                print 'LOSS:', loss
            logstats.update_summary_map(summary_map, {'loss': loss})
        return summary_map['loss']['mean']

    def _get_feed_dict(self, data):
        '''
        Take a data generator, return a feed_dict as input to the model.
        '''
        agent, kb, inputs, entities, targets, iswrite = data.next()
        if self.verbose:
            kb.dump()
            print 'INPUT:', map(self.data.vocab.to_word, list(inputs[0]))
            target_words = [self.data.vocab.to_word(t) if w else 'null' for t, w in zip(list(targets[0]), list(iswrite[0]))]
            print 'TARGET:', target_words
            #print 'WRITE:', iswrite
        feed_dict = {}
        self.model.update_feed_dict(feed_dict, inputs, None, targets=targets, kb=kb, entities=entities, iswrite=iswrite)
        return feed_dict

    def _learn_step(self, data, sess, summary_map):
        feed_dict = self._get_feed_dict(data)
        _, output, loss, gn, cgn = sess.run([self.train_op, self.model.outputs, self.model.loss, self.grad_norm, self.clipped_grad_norm], feed_dict=feed_dict)

        if self.verbose:
            pred = self.model.get_prediction(output)
            print 'PRED:', map(self.data.vocab.to_word, list(pred[0]))
            print 'LOSS:', loss

        logstats.update_summary_map(summary_map, \
                {'loss': loss, \
                'grad_norm': gn, \
                'clipped_grad_norm': cgn, \
                })

    def learn(self, args, config, ckpt=None, split='train'):
        assert args.optimizer in optim.keys()
        optimizer = optim[args.optimizer](args.learning_rate)

        # Gradient
        grads_and_vars = optimizer.compute_gradients(self.model.loss)
        if args.grad_clip > 0:
            min_grad, max_grad = -1.*args.grad_clip, args.grad_clip
            clipped_grads_and_vars = [(tf.clip_by_value(grad, min_grad, max_grad), var) for grad, var in grads_and_vars]
        else:
            clipped_grads_and_vars = grads_and_vars
        # TODO: clip has problem with indexedslices, don't use
        #self.clipped_grads = [grad for grad, var in clipped_grads_and_vars]
        #self.grads = [grad for grad, var in grads_and_vars]
        self.grad_norm = tf.global_norm([grad for grad, var in grads_and_vars])
        self.clipped_grad_norm = tf.global_norm([grad for grad, var in clipped_grads_and_vars])

        # Optimize
        self.train_op = optimizer.apply_gradients(clipped_grads_and_vars)

        # Training loop
        train_data = self.data.generator_train(split)
        # x2 because training from both agents' perspective
        num_per_epoch = args.num_per_epoch if args.num_per_epoch else self.data.num_examples[split] * 2
        step = 0
        saver = tf.train.Saver()
        save_path = os.path.join(args.checkpoint, 'tf_model.ckpt')
        best_saver = tf.train.Saver(max_to_keep=1)
        best_checkpoint = args.checkpoint+'-best'
        if not os.path.isdir(best_checkpoint):
            os.mkdir(best_checkpoint)
        best_save_path = os.path.join(best_checkpoint, 'tf_model.ckpt')
        best_bleu = -1

        with tf.Session(config=config) as sess:
            tf.initialize_all_variables().run()
            if args.init_from:
                saver.restore(sess, ckpt.model_checkpoint_path)
            summary_map = {}
            for epoch in xrange(args.max_epochs):
                print '================== Epoch %d ==================' % (epoch+1)
                for i in xrange(num_per_epoch):
                    start_time = time.time()
                    self._learn_step(train_data, sess, summary_map)
                    end_time = time.time()
                    logstats.update_summary_map(summary_map, \
                            {'time/batch': end_time - start_time})
                    step += 1
                    if step % args.print_every == 0:
                        print '{}/{} (epoch {}) {}'.format(i+1, num_per_epoch, epoch+1, logstats.summary_map_to_str(summary_map))
                        summary_map = {}  # Reset

                # Save model after each epoch
                print 'Save model checkpoint to', save_path
                saver.save(sess, save_path, global_step=epoch)

                # Evaluate on dev
                for eval_data in ('test',):
                    print '================== Eval %s ==================' % eval_data
                    bleu, ent_recall = self.test_bleu(sess, eval_data)
                    loss = self.test_loss(sess, eval_data)
                    if eval_data == 'test' and bleu > best_bleu:
                        print 'New best model'
                        best_bleu = bleu
                        best_saver.save(sess, best_save_path)
                    print 'bleu=%.4f entity_recall=%.4f loss=%.4f' % (bleu, ent_recall, loss)

