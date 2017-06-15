def add_data_generator_arguments(parser):
    from preprocess import add_preprocess_arguments
    from src.basic.scenario_db import add_scenario_arguments
    from src.basic.dataset import add_dataset_arguments

    add_scenario_arguments(parser)
    add_preprocess_arguments(parser)
    add_dataset_arguments(parser)

def get_data_generator(args, model_args, mappings, schema):
    from preprocess import DataGenerator, Preprocessor
    from src.basic.scenario_db import ScenarioDB
    from src.basic.negotiation.price_tracker import PriceTracker
    from src.basic.dataset import read_dataset
    from src.basic.util import read_json

    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))
    dataset = read_dataset(scenario_db, args)
    lexicon = PriceTracker()

    # Dataset
    preprocessor = Preprocessor(schema, lexicon, model_args.entity_encoding_form, model_args.entity_decoding_form, model_args.entity_target_form)
    if args.test:
        model_args.dropout = 0
        data_generator = DataGenerator(None, None, dataset.test_examples, preprocessor, schema, mappings)
    else:
        data_generator = DataGenerator(dataset.train_examples, dataset.test_examples, None, preprocessor, schema, mappings)

    return data_generator

def add_model_arguments(parser):
    from src.model.encdec import add_basic_model_arguments
    from src.model.sequence_embedder import add_sequence_embedder_arguments
    from price_predictor import add_price_predictor_arguments
    from context_embedder import add_context_embedder_arguments

    add_basic_model_arguments(parser)
    add_sequence_embedder_arguments(parser)
    add_price_predictor_arguments(parser)
    add_context_embedder_arguments(parser)

def build_model(schema, mappings, args):
    import tensorflow as tf
    from src.model.word_embedder import WordEmbedder
    from src.model.encdec import BasicEncoder, BasicDecoder, Sampler
    from price_predictor import PricePredictor
    from encdec import BasicEncoderDecoder, PriceDecoder, ContextDecoder, AttentionDecoder
    from context_embedder import ContextEmbedder
    from preprocess import markers
    from src.model.sequence_embedder import get_sequence_embedder

    tf.reset_default_graph()
    tf.set_random_seed(args.random_seed)

    with tf.variable_scope('GlobalDropout'):
        if args.test:
            keep_prob = tf.constant(1.)
        else:
            keep_prob = tf.constant(1. - args.dropout)

    vocab = mappings['vocab']
    pad = vocab.to_ind(markers.PAD)

    with tf.variable_scope('EncoderWordEmbedder'):
        encoder_word_embedder = WordEmbedder(vocab.size, args.word_embed_size, pad)
    with tf.variable_scope('DecoderWordEmbedder'):
        decoder_word_embedder = WordEmbedder(vocab.size, args.word_embed_size, pad)

    if args.decoding[0] == 'sample':
        sample_t = float(args.decoding[1])
        sampler = Sampler(sample_t)
    else:
        raise('Unknown decoding method')

    re_encode = args.re_encode

    opts = vars(args)
    opts['vocab_size'] = vocab.size
    opts['keep_prob'] = keep_prob
    opts['embed_size'] = args.rnn_size
    encoder_seq_embedder = get_sequence_embedder(args.encoder, **opts)
    decoder_seq_embedder = get_sequence_embedder(args.decoder, **opts)

    if args.context is not None:
        context_opts = dict(opts)
        context_opts['vocab_size'] = mappings['kb_vocab'].size
        context_opts['embed_size'] = args.context_size
        with tf.variable_scope('ContextWordEmbedder'):
            context_word_embedder = WordEmbedder(context_opts['vocab_size'], context_opts['embed_size'], pad)
        context_seq_embedder = get_sequence_embedder(args.context_encoder, **context_opts)
        context_embedder = ContextEmbedder(mappings['cat_vocab'].size, context_word_embedder, context_seq_embedder, pad)

    if args.model == 'encdec':
        encoder = BasicEncoder(encoder_word_embedder, encoder_seq_embedder, pad, keep_prob)
        if args.context is not None:
            #decoder = ContextDecoder(decoder_word_embedder, decoder_seq_embedder, context_embedder, args.context, pad, keep_prob, vocab.size, sampler, args.sampled_loss)
            decoder = AttentionDecoder(decoder_word_embedder, decoder_seq_embedder, pad, keep_prob, vocab.size, sampler, args.sampled_loss, context_embedder=context_embedder)
        else:
            #decoder = BasicDecoder(decoder_word_embedder, decoder_seq_embedder, pad, keep_prob, vocab.size, sampler, args.sampled_loss)
            decoder = AttentionDecoder(decoder_word_embedder, decoder_seq_embedder, pad, keep_prob, vocab.size, sampler, args.sampled_loss)
        if args.predict_price:
            price_predictor = PricePredictor(args.price_predictor_hidden_size, 1+2*args.price_hist_len)
            decoder = PriceDecoder(decoder, price_predictor)
        model = BasicEncoderDecoder(encoder, decoder, pad, re_encode=re_encode)
    else:
        raise ValueError('Unknown model')
    return model

