'''
Main learning loop (using keras models).
'''

def add_learner_arguments(parser):
    parser.add_argument('--optimizer', default='adagrad', help='Optimization method')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--init-from', help='Initial parameters')

class Learner(object):
    def __init__(self, data, model):
        self.data = data  # DataGenerator object
        self.model = model

    def learn(self, args):
        # Compile keras model
        self.model.model.compile(optimizer=args.optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        self.model.model.fit_generator(self.data.generator('train'),
                        samples_per_epoch=len(self.data.examples['train']),
                        nb_epoch=args.num_epochs,
                        validation_data=self.data.generator('dev'),
                        nb_val_samples=len(self.data.examples['dev']),
                        verbose=2)

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError
