'''
Main learning loop.
'''

import tensorflow as tf
from itertools import izip

def add_learner_arguments(parser):
    parser.add_argument('--optimizer', default='adagrad', help='Optimization method')
    parser.add_argument('--grad-clip', type=list, help='Min and max values of gradients')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--max-epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--num-per-epoch', type=int, help='Number of examples per epoch')
    parser.add_argument('--print-every', type=int, default=1, help='Number of examples between printing training loss')
    parser.add_argument('--init-from', help='Initial parameters')
    parser.add_argument('--model-dir', help='Directory to save learned models')

optim = {'adagrad': tf.train.AdagradOptimizer,
         'sgd': tf.train.GradientDescentOptimizer,
        }

class Learner(object):
    def __init__(self, data, model):
        self.data = data  # DataGenerator object
        self.model = model

    def learn(self, args):
        tvars = tf.trainable_variables()

        # Gradient
        grads = tf.gradients(self.model.loss, tvars)
        if args.grad_clip:
            min_grad, max_grad = args.grad_clip
            grads = tf.clip_by_value(grads, min_grad, max_grad)

        # Optimize
        assert args.optimizer in optim.keys()
        optimizer = optim[args.optimizer](args.learning_rate)
        train_op = optimizer.apply_gradients(izip(grads, tvars))

        # Training loop
        train_data = self.data.generator('train')
        num_per_epoch = args.num_per_epoch if args.num_per_epoch else len(self.data.examples['train'])
        step = 0
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            for epoch in xrange(args.max_epochs):
                for i in xrange(num_per_epoch):
                    inputs, iswrite, targets = train_data.next()
                    feed_dict = {self.model.input_data: inputs,
                            self.model.input_iswrite: iswrite,
                            self.model.targets: targets}
                    _, loss = sess.run([train_op, self.model.loss], feed_dict=feed_dict)
                    step += 1
                    if step % args.print_every == 0:
                        print '{}/{} (epoch {}), train_loss={:.2f}'.format(i+1, num_per_epoch, epoch+1, loss)
