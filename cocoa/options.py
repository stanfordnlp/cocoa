
def add_rulebased_arguments(parser):
    parser.add_argument('--templates', help='Path to templates (.pkl)')
    parser.add_argument('--policy', help='Path to manager model (.pkl)')

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
