import onmt

def add_data_generator_arguments(parser):
    from preprocess import add_preprocess_arguments
    from cocoa.core.scenario_db import add_scenario_arguments
    from cocoa.core.dataset import add_dataset_arguments

    add_scenario_arguments(parser)
    add_preprocess_arguments(parser)
    add_dataset_arguments(parser)

def get_data_generator(args, model_args, schema, test=False):
    from cocoa.core.scenario_db import ScenarioDB
    from cocoa.core.dataset import read_dataset
    from cocoa.core.util import read_json

    from core.scenario import Scenario
    from core.lexicon import Lexicon
    from preprocess import DataGenerator, LMDataGenerator, Preprocessor
    import os.path

    # TODO: move this to dataset
    dataset = read_dataset(args, Scenario)

    # Model config tells data generator which batcher to use
    model_config = {}
    add_ground_truth = False
    mappings_path = model_args.mappings

    # settings for other models
    retriever = None
    trie_path = None

    lexicon = Lexicon(['book', 'hat', 'ball'])
    preprocessor = Preprocessor(schema, lexicon, model_args.entity_encoding_form,
        model_args.entity_decoding_form, model_args.entity_target_form,
        model=model_args.model)

    if test:
        model_args.dropout = 0
        train, dev, test = None, None, dataset.test_examples
    else:
        train, dev, test = dataset.train_examples, dataset.test_examples, None
    data_generator = DataGenerator(train, dev, test, preprocessor, args, schema, mappings_path,
        retriever=retriever, cache=args.cache, ignore_cache=args.ignore_cache,
        candidates_path=args.candidates_path, num_context=model_args.num_context,
        trie_path=trie_path, batch_size=args.batch_size,
        model=model_args.model, add_ground_truth=add_ground_truth)

    return data_generator

def check_model_args(args):
    if args.pretrained_wordvec:
        if isinstance(args.pretrained_wordvec, list):
            pretrained = args.pretrained_wordvec[0]
        else:
            pretrained = args.pretrained_wordvec
        with open(pretrained, 'r') as fin:
            pretrained_word_embed_size = len(fin.readline().strip().split()) - 1
        assert pretrained_word_embed_size == args.word_embed_size

        if args.context and args.context_encoder == 'bow':
            assert pretrained_word_embed_size == args.context_size

    if args.decoder == 'rnn-attn':
        assert args.attention_memory is not None

    if args.num_context > 0:
        assert not args.stateful

    assert args.temperature >= 0

def make_model_mappings(model, mappings):
    if model == 'seq2lf':
        mappings['src_vocab'] = mappings['utterance_vocab']
        mappings['tgt_vocab'] = mappings['lf_vocab']
    else:
        mappings['src_vocab'] = mappings['utterance_vocab']
        mappings['tgt_vocab'] = mappings['utterance_vocab']
    return mappings

def build_optim(opt, model, checkpoint):
    if opt.train_from:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    else:
        print('Making optimizer for training.')
        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            model_size=opt.rnn_size)

    optim.set_parameters(model.parameters())

    return optim
