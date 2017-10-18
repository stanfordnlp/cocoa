def add_data_generator_arguments(parser):
    from preprocess import add_preprocess_arguments
    from cocoa.core.scenario_db import add_scenario_arguments
    from cocoa.core.mutualfriends.lexicon import add_lexicon_arguments
    from cocoa.core.dataset import add_dataset_arguments

    add_scenario_arguments(parser)
    add_lexicon_arguments(parser)
    add_preprocess_arguments(parser)
    add_dataset_arguments(parser)

def get_data_generator(args, model_args, mappings, schema):
    from preprocess import DataGenerator, Preprocessor
    from cocoa.core.scenario_db import ScenarioDB
    from cocoa.core.mutualfriends.lexicon import Lexicon
    from cocoa.core.dataset import read_dataset
    from cocoa.core.util import read_json
    import time

    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))
    dataset = read_dataset(scenario_db, args)
    print 'Building lexicon...'
    start = time.time()
    lexicon = Lexicon(schema, args.learned_lex, stop_words=args.stop_words)
    print '%.2f s'% (time.time() - start)

    # Dataset
    use_kb = False if model_args.model == 'encdec' else True
    copy = True if model_args.model == 'attn-copy-encdec' else False
    if model_args.model == 'attn-copy-encdec':
        model_args.entity_target_form = 'graph'
    preprocessor = Preprocessor(schema, lexicon, model_args.entity_encoding_form, model_args.entity_decoding_form, model_args.entity_target_form, model_args.prepend)
    if args.test:
        model_args.dropout = 0
        data_generator = DataGenerator(None, None, dataset.test_examples, preprocessor, schema, model_args.num_items, mappings, use_kb, copy)
    else:
        data_generator = DataGenerator(dataset.train_examples, dataset.test_examples, None, preprocessor, schema, model_args.num_items, mappings, use_kb, copy)

    return data_generator

def add_model_arguments(parser):
    from cocoa.model.encdec import add_model_arguments
    from rnn_cell import add_attention_arguments
    from graph import add_graph_arguments
    from graph_embedder import add_graph_embed_arguments

    add.core.model_arguments(parser)
    add_attention_arguments(parser)
    add_graph_arguments(parser)
    add_graph_embed_arguments(parser)
    parser.add_argument('--separate-utterance-embedding', default=False, action='store_true', help='Re-encode the utterance (to be added to the graph embedding) starting from zero init_state')
    parser.add_argument('--gated-copy', default=False, action='store_true', help='Use gating function for copy')
    parser.add_argument('--sup-gate', default=False, action='store_true', help='Supervise copy gate')
    parser.add_argument('--preselect', default=False, action='store_true', help='Pre-select entities before decoding')
    parser.add_argument('--node-embed-in-rnn-inputs', default=False, action='store_true', help='Add node embedding of entities as inputs to the RNN')
    parser.add_argument('--no-graph-update', default=False, action='store_true', help='Do not update the KB graph during the dialogue')

# TODO: factor this
def build_model(schema, mappings, args):
    import tensorflow as tf
    from cocoa.model.word_embedder import WordEmbedder
    from cocoa.model.encdec import BasicEncoder, BasicDecoder, Sampler
    from encdec import GraphEncoder, GraphDecoder, CopyGraphDecoder, PreselectCopyGraphDecoder, GatedCopyGraphDecoder, BasicEncoderDecoder, GraphEncoderDecoder
    from preprocess import markers
    from graph_embedder import GraphEmbedder
    from graph_embedder_config import GraphEmbedderConfig
    from graph import Graph, GraphMetadata

    tf.reset_default_graph()
    tf.set_random_seed(args.random_seed)

    vocab = mappings['vocab']
    pad = vocab.to_ind(markers.PAD)
    select = vocab.to_ind(markers.SELECT)
    with tf.variable_scope('EncoderWordEmbedder'):
        encoder_word_embedder = WordEmbedder(vocab.size, args.word_embed_size, pad)
    with tf.variable_scope('DecoderWordEmbedder'):
        decoder_word_embedder = WordEmbedder(vocab.size, args.word_embed_size, pad)

    if args.decoding[0] == 'sample':
        sample_t = float(args.decoding[1])
        sample_select = None if len(args.decoding) < 3 or args.decoding[2] != 'select' else select
        sampler = Sampler(sample_t, sample_select)
    else:
        raise('Unknown decoding method')

    separate_utterance_embedding = args.separate_utterance_embedding
    re_encode = args.re_encode

    update_graph = (not args.no_graph_update)
    node_embed_in_rnn_inputs = args.node_embed_in_rnn_inputs

    if args.model == 'encdec':
        encoder = BasicEncoder(args.rnn_size, args.rnn_type, args.num_layers, args.dropout)
        decoder = BasicDecoder(args.rnn_size, vocab.size, args.rnn_type, args.num_layers, args.dropout, sampler=sampler)
        model = BasicEncoderDecoder(encoder_word_embedder, decoder_word_embedder, encoder, decoder, pad, select, re_encode=re_encode)
    elif args.model == 'attn-encdec' or args.model == 'attn-copy-encdec':
        max_degree = args.num_items + len(schema.attributes)
        utterance_size = args.rnn_size
        graph_metadata = GraphMetadata(schema, mappings['entity'], mappings['relation'], utterance_size, args.max_num_entities, max_degree=max_degree, entity_hist_len=args.entity_hist_len, max_num_items=args.num_items)
        graph_embedder_config = GraphEmbedderConfig(args.node_embed_size, args.edge_embed_size, graph_metadata, entity_embed_size=args.entity_embed_size, use_entity_embedding=args.use_entity_embedding, mp_iters=args.mp_iters, decay=args.utterance_decay, msg_agg=args.msg_aggregation, learned_decay=args.learned_utterance_decay)
        Graph.metadata = graph_metadata
        graph_embedder = GraphEmbedder(graph_embedder_config)
        encoder = GraphEncoder(args.rnn_size, graph_embedder, rnn_type=args.rnn_type, num_layers=args.num_layers, dropout=args.dropout, update_graph=update_graph, node_embed_in_rnn_inputs=node_embed_in_rnn_inputs, separate_utterance_embedding=separate_utterance_embedding)
        if args.model == 'attn-encdec':
            decoder = GraphDecoder(args.rnn_size, vocab.size, graph_embedder, rnn_type=args.rnn_type, num_layers=args.num_layers, checklist=(not args.no_checklist), dropout=args.dropout, sampler=sampler, update_graph=update_graph, node_embed_in_rnn_inputs=node_embed_in_rnn_inputs, separate_utterance_embedding=separate_utterance_embedding, encoder=encoder)
        elif args.model == 'attn-copy-encdec':
            if args.gated_copy:
                decoder = GatedCopyGraphDecoder(args.rnn_size, vocab.size, graph_embedder, rnn_type=args.rnn_type, num_layers=args.num_layers, checklist=(not args.no_checklist), dropout=args.dropout, sampler=sampler, update_graph=update_graph, node_embed_in_rnn_inputs=node_embed_in_rnn_inputs, separate_utterance_embedding=separate_utterance_embedding, encoder=encoder)
                sup_gate = args.sup_gate
            else:
                if args.preselect:
                    decoder = PreselectCopyGraphDecoder(args.rnn_size, vocab.size, graph_embedder, rnn_type=args.rnn_type, num_layers=args.num_layers, checklist=(not args.no_checklist), dropout=args.dropout, sampler=sampler, update_graph=update_graph, node_embed_in_rnn_inputs=node_embed_in_rnn_inputs, separate_utterance_embedding=separate_utterance_embedding, encoder=encoder)
                else:
                    decoder = CopyGraphDecoder(args.rnn_size, vocab.size, graph_embedder, rnn_type=args.rnn_type, num_layers=args.num_layers, checklist=(not args.no_checklist), dropout=args.dropout, sampler=sampler, update_graph=update_graph, node_embed_in_rnn_inputs=node_embed_in_rnn_inputs, separate_utterance_embedding=separate_utterance_embedding, encoder=encoder)
                sup_gate = False
        model = GraphEncoderDecoder(encoder_word_embedder, decoder_word_embedder, graph_embedder, encoder, decoder, pad, select, re_encode=re_encode, sup_gate=sup_gate)
    else:
        raise ValueError('Unknown model')
    return model

