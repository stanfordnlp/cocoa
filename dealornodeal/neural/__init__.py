import onmt

def get_data_generator(args, model_args, schema, test=False):
    from cocoa.core.scenario_db import ScenarioDB
    from cocoa.core.dataset import read_dataset
    from cocoa.core.util import read_json

    from core.scenario import Scenario
    from core.lexicon import Lexicon
    from preprocess import DataGenerator, Preprocessor
    import os.path

    # TODO: move this to dataset
    dataset = read_dataset(args, Scenario)

    mappings_path = model_args.mappings

    lexicon = Lexicon(schema.values['item'])
    preprocessor = Preprocessor(schema, lexicon, model_args.entity_encoding_form,
        model_args.entity_decoding_form, model_args.entity_target_form,
        model=model_args.model)

    if test:
        model_args.dropout = 0
        train, dev, test = None, None, dataset.test_examples
    else:
        train, dev, test = dataset.train_examples, dataset.test_examples, None
    data_generator = DataGenerator(train, dev, test, preprocessor, args, schema, mappings_path,
        cache=args.cache, ignore_cache=args.ignore_cache,
        num_context=model_args.num_context,
        batch_size=args.batch_size,
        model=model_args.model)

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
    print('Making optimizer for training.')
    optim = onmt.Optim(
        opt.optim, opt.learning_rate, opt.max_grad_norm,
        model_size=opt.rnn_size)

    optim.set_parameters(model.parameters())

    return optim
