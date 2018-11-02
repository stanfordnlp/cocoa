import cocoa.options

# =============== data ===============
def add_preprocess_arguments(parser):
    parser.add_argument('--entity-encoding-form', choices=['canonical', 'type'], default='canonical', help='Input entity form to the encoder')
    parser.add_argument('--entity-decoding-form', choices=['canonical', 'type'], default='canonical', help='Input entity form to the decoder')
    parser.add_argument('--entity-target-form', choices=['canonical', 'type'], default='canonical', help='Output entity form to the decoder')
    parser.add_argument('--cache', default='.cache', help='Path to cache for preprocessed batches')
    parser.add_argument('--ignore-cache', action='store_true', help='Ignore existing cache')
    parser.add_argument('--mappings', help='Path to vocab mappings')

def add_data_generator_arguments(parser):
    cocoa.options.add_scenario_arguments(parser)
    cocoa.options.add_dataset_arguments(parser)
    add_preprocess_arguments(parser)


# =============== model ===============
def add_rl_arguments(parser):
    cocoa.options.add_rl_arguments(parser)
    parser.add_argument('--reward', choices=['margin', 'length', 'fair'],
            help='Which reward function to use')

def add_model_arguments(parser):
    from onmt.modules.SRU import CheckSRU
    group = parser.add_argument_group('Model')
    group.add_argument('--encoder-type', type=str, default='rnn',
                       choices=['rnn', 'brnn', 'transformer', 'cnn'],
                       help="""Type of encoder layer to use. Non-RNN layers
                       are experimental. Options are
                       [rnn|brnn|mean|transformer|cnn].""")
    group.add_argument('--context-embedder-type', type=str, default='mean',
                       choices=['rnn', 'mean', 'brnn'],
                       help="Encoder to use for embedding prev turns context")
    group.add_argument('--decoder-type', type=str, default='rnn',
                       choices=['rnn', 'transformer', 'cnn'],
                       help="""Type of decoder layer to use. Non-RNN layers
                       are experimental. Options are [rnn|transformer|cnn].""")
    group.add_argument('--copy-attn', action="store_true",
                       help='Train copy attention layer.')
    group.add_argument('--layers', type=int, default=-1,
                       help='Number of layers in enc/dec.')
    group.add_argument('--enc-layers', type=int, default=1,
                       help='Number of layers in the encoder')
    group.add_argument('--dec-layers', type=int, default=1,
                       help='Number of layers in the decoder')
    group.add_argument('--rnn-type', type=str, default='LSTM',
                       choices=['LSTM', 'GRU', 'SRU'], action=CheckSRU,
                       help="""The gate type to use in the RNNs""")

    # Hidden sizes and dimensions
    group.add_argument('--word-vec-size', type=int, default=256,
                       help='Word embedding size for src and tgt.')
    group.add_argument('--sel-hid-size', type=int, default=128,
                       help='Size of hidden state for selectors')
    group.add_argument('--kb-embed-size', type=int, default=64,
                       help='Size of vocab embeddings and output for kb scenario')
    group.add_argument('--rnn-size', type=int, default=256,
                       help='Size of rnn hidden states')

    # Our custom flags
    group.add_argument('--input-feed', action='store_true',
                       help="""Feed the context vector at each time step as
                       additional input (via concatenation with the word
                       embeddings) to the decoder.""")
    group.add_argument('--global-attention', type=str, default='multibank_general',
                       choices=['dot', 'general', 'mlp',
                       'multibank_dot', 'multibank_general', 'multibank_mlp'],
                       help="""The attention type to use: dotprod or general (Luong)
                       or MLP (Bahdanau), prepend multibank to add context""")
    group.add_argument('--model', type=str, default='seq2seq',
                       choices=['seq2seq', 'seq2lf', 'sum2sum', 'sum2seq', \
                       'lf2lf', 'lflm', 'seq_select'], help='Model type')
    group.add_argument('--num-context', type=int, default=2,
                       help='Number of sentences to consider as dialogue context (in addition to the encoder input)')
    group.add_argument('--stateful', action='store_true',
                       help='Whether to pass on the hidden state throughout the dialogue encoding/decoding process')
    group.add_argument('--share-embeddings', action='store_true',
                       help='Share source and target vocab embeddings')


# =============== system ===============
def add_neural_system_arguments(parser):
    cocoa.options.add_generator_arguments(parser)

def add_system_arguments(parser):
    parser.add_argument('--mappings', default='.', help='Directory to save mappings/vocab')
    add_neural_system_arguments(parser)
    cocoa.options.add_rulebased_arguments(parser)

def add_hybrid_system_arguments(parser):
    cocoa.options.add_rulebased_arguments(parser)
    add_neural_system_arguments(parser)


# =============== web ===============
def add_website_arguments(parser):
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to start server on')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host IP address to run app on. Defaults to localhost.')
    parser.add_argument('--config', type=str, default='app_params.json',
                        help='Path to JSON file containing configurations for website')
    parser.add_argument('--output', type=str,
                        default="web_output/{}".format(datetime.now().strftime("%Y-%m-%d")),
                        help='Name of directory for storing website output (debug and error logs, chats, '
                             'and database). Defaults to a web_output/current_date, with the current date formatted as '
                             '%%Y-%%m-%%d. '
                             'If the provided directory exists, all data in it is overwritten unless the '
                             '--reuse parameter is provided.')
    parser.add_argument('--reuse', action='store_true', help='If provided, reuses the existing database file in the '
                                                             'output directory.')
