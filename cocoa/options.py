# =============== core ===============
def add_dataset_arguments(parser):
    parser.add_argument('--train-examples-paths', nargs='*', default=[],
        help='Input training examples')
    parser.add_argument('--test-examples-paths', nargs='*', default=[],
        help='Input test examples')
    parser.add_argument('--train-max-examples', type=int,
        help='Maximum number of training examples')
    parser.add_argument('--test-max-examples', type=int,
        help='Maximum number of test examples')
    parser.add_argument('--eval-examples-paths', nargs='*', default=[],
        help='Path to multi-response evaluation files')

def add_scenario_arguments(parser):
    parser.add_argument('--schema-path', help='Input path that describes the schema of the domain')
    parser.add_argument('--scenarios-path', help='Output path for the scenarios generated')


# =============== model ===============
def add_logging_arguments(parser):
    group = parser.add_argument_group('Logging')
    group.add_argument('--report-every', type=int, default=5,
                       help="Print stats at this many batch intervals")
    group.add_argument('--model-filename', default='model',
                       help="""Model filename (the model will be saved as
                       <filename>_acc_ppl_e.pt where ACC is accuracy, PPL is
                       the perplexity and E is the epoch""")
    group.add_argument('--model-path', default='data/checkpoints',
                       help="""Which file the model checkpoints will be saved""")
    group.add_argument('--start-checkpoint-at', type=int, default=0,
                       help="""Start checkpointing every epoch after and including
                       this epoch""")
    group.add_argument('--best-only', action='store_true',
                       help="Only store the best checkpoint")

def add_trainer_arguments(parser):
    group = parser.add_argument_group('Training')

    # Initialization
    group.add_argument('--pretrained-wordvec', nargs='+', default=['', ''],
                       help="""If a valid path is specified, then this will load
                       pretrained word embeddings, if list contains two embeddings,
                       then the second one is for item title and description""")
    group.add_argument('--param-init', type=float, default=0.1,
                       help="""Parameters are initialized over uniform distribution
                       with support (-param_init, param_init).
                       Use 0 to not use initialization""")
    group.add_argument('--fix-pretrained-wordvec',
                       action='store_true',
                       help="Fix pretrained word embeddings.")

    # Optimization
    group.add_argument('--batch-size', type=int, default=64,
                       help='Maximum batch size for training')
    # group.add_argument('--batches_per_epoch', type=int, default=10,
    #                    help='Data comes from a generator, which is unlimited, so we need to set some artificial limit.')
    group.add_argument('--epochs', type=int, default=14,
                       help='Number of training epochs')
    group.add_argument('--optim', default='sgd', help="""Optimization method.""",
                       choices=['sgd', 'adagrad', 'adadelta', 'adam'])
    group.add_argument('--max-grad-norm', type=float, default=5,
                       help="""If the norm of the gradient vector exceeds this,
                       renormalize it to have the norm equal to max_grad_norm""")
    group.add_argument('--dropout', type=float, default=0.3,
                       help="Dropout probability; applied in LSTM stacks.")
    group.add_argument('--learning-rate', type=float, default=1.0,
                       help="""Starting learning rate. Recommended settings:
                       sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")
    group.add_argument('--gpuid', default=[], nargs='+', type=int,
                       help="Use CUDA on the listed devices.")
    group.add_argument('-seed', type=int, default=-1,
                       help="""Random seed used for the experiments reproducibility.""")
    group.add_argument('--label-smoothing', type=float, default=0.0,
                       help="""Label smoothing value epsilon.
                       Probabilities of all non-true labels will be smoothed
                       by epsilon / (vocab_size - 1). Set to zero to turn off
                       label smoothing. For more detailed information, see:
                       https://arxiv.org/abs/1512.00567""")

    # Logging
    add_logging_arguments(parser)

def add_rl_arguments(parser):
    group = parser.add_argument_group('Reinforce')
    group.add_argument('--max-turns', default=100, type=int, help='Maximum number of turns')
    group.add_argument('--num-dialogues', default=10000, type=int,
            help='Number of dialogues to generate/train')
    group.add_argument('--discount-factor', default=1.0, type=float,
            help='Amount to discount the reward for each timestep when \
            calculating the value, usually written as gamma')
    group.add_argument('--verbose', default=False, action='store_true',
            help='Whether or not to have verbose prints')

    group = parser.add_argument_group('Training')
    group.add_argument('--optim', default='sgd', help="""Optimization method.""",
                       choices=['sgd', 'adagrad', 'adadelta', 'adam'])
    group.add_argument('--epochs', type=int, default=14,
                       help='Number of training epochs')
    group.add_argument('--batch-size', type=int, default=64,
                       help='Maximum batch size for training')
    group.add_argument('--max-grad-norm', type=float, default=5,
                       help="""If the norm of the gradient vector exceeds this,
                       renormalize it to have the norm equal to max_grad_norm""")
    group.add_argument('--learning-rate', type=float, default=1.0,
                       help="""Starting learning rate. Recommended settings:
                       sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")

    # Logging
    add_logging_arguments(parser)


def add_generator_arguments(parser):
    """Arguments for generating text from pretrained neural models.
    """
    parser.add_argument('--checkpoint',
                       help='Path to checkpoint')

    group = parser.add_argument_group('Beam')
    group.add_argument('--beam-size',  type=int, default=5,
                       help='Beam size')
    group.add_argument('--min-length', type=int, default=1,
                       help='Minimum prediction length')
    group.add_argument('--max-length', type=int, default=50,
                       help='Maximum prediction length.')
    group.add_argument('--n-best', type=int, default=1,
                help="""If verbose is set, will output the n_best decoded sentences""")
    group.add_argument('--alpha', type=float, default=0.5,
                help="""length penalty parameter (higher = longer generation)""")

    group = parser.add_argument_group('Sample')
    group.add_argument('--sample', action="store_true",
                       help='Sample instead of beam search')
    group.add_argument('--temperature', type=float, default=1,
                help="""Sample temperature""")

    group = parser.add_argument_group('Efficiency')
    group.add_argument('--batch-size', type=int, default=30,
                       help='Batch size')
    group.add_argument('--gpuid', default=[], nargs='+', type=int,
                       help="Use CUDA on the listed devices.")

    group = parser.add_argument_group('Logging')
    group.add_argument('--verbose', action="store_true",
                       help='Print scores and predictions for each sentence')


# =============== system ===============
def add_rulebased_arguments(parser):
    parser.add_argument('--templates', help='Path to templates (.pkl)')
    parser.add_argument('--policy', help='Path to manager model (.pkl)')

