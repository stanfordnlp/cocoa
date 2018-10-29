from cocoa.options import add_generator_arguments, add_rulebased_arguments

def add_price_tracker_arguments(parser):
    parser.add_argument('--price-tracker-model', help='Path to price tracker model')

def add_neural_system_arguments(parser):
    add_generator_arguments(parser)


