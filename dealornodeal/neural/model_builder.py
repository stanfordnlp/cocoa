"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn
import onmt
import onmt.io
import onmt.Models
import onmt.modules
from onmt.modules import Embeddings, ImageEncoder, CopyGenerator

from cocoa.neural.models import MeanEncoder, StdRNNEncoder, StdRNNDecoder, \
              MultiAttnDecoder, NMTModel, MultiAttnConcatDecoder
from models import NegotiationModel

from cocoa.io.utils import read_pickle
from onmt.Utils import use_gpu

from symbols import markers
from neural import make_model_mappings


def build_model(model_opt, opt, fields, checkpoint):
    print('Building model...')
    model = onmt.ModelConstructor.make_base_model(model_opt, fields,
                                                  use_gpu(opt), checkpoint)
    if len(opt.gpuid) > 1:
        print('Multi gpu training: ', opt.gpuid)
        model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
    print(model)

    return model


def make_embeddings(opt, word_dict, for_encoder=True, embed_type=None):
    """
    Make an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocabulary): words dictionary.
        for_encoder(bool): make Embeddings for encoder or decoder?
    """
    embedding_dim = opt.kb_embed_size if embed_type == "kb" else opt.word_vec_size

    word_padding_idx = word_dict.to_ind(markers.PAD)
    num_word_embeddings = len(word_dict)

    return Embeddings(word_vec_size=embedding_dim,
                      position_encoding=False,
                      dropout=opt.dropout,
                      word_padding_idx=word_padding_idx,
                      word_vocab_size=num_word_embeddings)


def make_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    if opt.encoder_type == "transformer":
        return TransformerEncoder(opt.enc_layers, opt.rnn_size,
                                  opt.dropout, embeddings)
    elif opt.encoder_type == "cnn":
        return CNNEncoder(opt.enc_layers, opt.rnn_size,
                          opt.cnn_kernel_width,
                          opt.dropout, embeddings)
    elif opt.encoder_type == "mean":
        return MeanEncoder(opt.enc_layers, embeddings)
    else:
        # "rnn" or "brnn"
        bidirectional = True if opt.encoder_type == 'brnn' else False
        return StdRNNEncoder(opt.rnn_type, bidirectional, opt.enc_layers,
                        opt.rnn_size, opt.dropout, embeddings)

def make_context_embedder(opt, embeddings, embed_type='utterance'):
    """
    Various context embedder dispatcher function. See make_encoder for options
    embed_type (:str:): either dialogue, kb (for title and desc) or category
    """
    if opt.context_embedder_type == "mean":
        num_layers = 1 if embed_type == "scenario" else opt.enc_layers
        return MeanEncoder(num_layers, embeddings, embed_type)
    else:
        # "rnn" or "brnn"
        bidirectional = True if opt.context_embedder_type == 'brnn' else False
        return StdRNNEncoder(opt.rnn_type, bidirectional, opt.enc_layers,
                        opt.rnn_size, opt.dropout, embeddings, embed_type, False)

def make_decoder(opt, embeddings, tgt_dict):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    bidirectional = True if opt.encoder_type == 'brnn' else False
    pad = tgt_dict.to_ind(markers.PAD)
    if "multibank" in opt.global_attention:
        #return MultiAttnDecoder(opt.rnn_type, bidirectional,
        #                     opt.dec_layers, opt.rnn_size,
        #                     attn_type=opt.global_attention,
        #                     dropout=opt.dropout,
        #                     embeddings=embeddings,
        #                     pad=pad)
        # TODO: hacky memory_sizes
        return MultiAttnConcatDecoder(opt.rnn_type, bidirectional,
                             opt.dec_layers, opt.rnn_size,
                             (opt.rnn_size, opt.word_vec_size, opt.kb_embed_size,),
                             attn_type=opt.global_attention,
                             dropout=opt.dropout,
                             embeddings=embeddings,
                             pad=pad)
    elif opt.decoder_type == "transformer":
        return TransformerDecoder(opt.dec_layers, opt.rnn_size,
                                  opt.global_attention, opt.copy_attn,
                                  opt.dropout, embeddings)
    elif opt.decoder_type == "cnn":
        return CNNDecoder(opt.dec_layers, opt.rnn_size,
                          opt.global_attention, opt.copy_attn,
                          opt.cnn_kernel_width, opt.dropout,
                          embeddings)
    elif opt.input_feed:
        return InputFeedRNNDecoder(opt.rnn_type, bidirectional,
                                   opt.dec_layers, opt.rnn_size,
                                   attn_type=opt.global_attention,
                                   dropout=opt.dropout,
                                   embeddings=embeddings)
    else:
        return StdRNNDecoder(opt.rnn_type, bidirectional,
                             opt.dec_layers, opt.rnn_size,
                             attn_type=opt.global_attention,
                             dropout=opt.dropout,
                             embeddings=embeddings,
                             pad=pad)

def load_test_model(model_path, opt, dummy_opt):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

    model_opt = checkpoint['opt']
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]

    mappings = read_pickle('{}/vocab.pkl'.format(model_opt.mappings))
    mappings = make_model_mappings(model_opt.model, mappings)

    model = make_base_model(model_opt, mappings, use_gpu(opt), checkpoint)
    model.eval()
    model.generator.eval()
    return mappings, model, model_opt


# TODO: remove selectors. we don't use them. but keep them for now
# due to match parameters in existing models
def make_selectors(opt, kb_dict):
    selectors = {}
    input_size = opt.rnn_size + (6 * opt.kb_embed_size)
    selectors["enc"] = nn.GRU( input_size=opt.rnn_size,
            hidden_size=opt.kb_embed_size, bias=True, bidirectional=True)
    selectors["dec"] = nn.Sequential(
       nn.Linear(11 * opt.rnn_size, opt.rnn_size),  # 2816 to 256
       nn.Tanh(),
       nn.Linear(opt.rnn_size, opt.sel_hid_size),   # 256 to 128
       nn.Tanh(),
       nn.Linear(opt.sel_hid_size, 6 * kb_dict.size)  # 128 to 60
    )
    return selectors

def make_base_model(model_opt, mappings, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    # Make encoder.
    src_dict = mappings['src_vocab']
    src_embeddings = make_embeddings(model_opt, src_dict)
    encoder = make_encoder(model_opt, src_embeddings)

    # Make context embedder.
    if model_opt.num_context > 0:
        context_dict = mappings['utterance_vocab']
        context_embeddings = make_embeddings(model_opt, context_dict)
        context_embedder = make_context_embedder(model_opt, context_embeddings)

    kb_dict = mappings['kb_vocab']
    # kb_embeddings = make_embeddings(model_opt, kb_dict, True, 'kb')
    # kb_embedder = make_context_embedder(model_opt, kb_embeddings, 'scenario')
    scene_settings = (kb_dict.size, model_opt.kb_embed_size)

    # Make decoder.
    tgt_dict = mappings['tgt_vocab']
    tgt_embeddings = make_embeddings(model_opt, tgt_dict)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')

        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    selectors = make_selectors(model_opt, kb_dict)
    dropout = nn.Dropout(model_opt.dropout)
    decoder = make_decoder(model_opt, tgt_embeddings, tgt_dict)

    if "multibank" in model_opt.global_attention:
        model = NegotiationModel(encoder, decoder, context_embedder,
                selectors, scene_settings, dropout,
                stateful=model_opt.stateful)
    else:
        model = NMTModel(encoder, decoder, stateful=model_opt.stateful)

    # Make Generator.
    if not model_opt.copy_attn:
        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, len(tgt_dict)),
            nn.LogSoftmax())
        if model_opt.share_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        generator = CopyGenerator(model_opt.rnn_size,
                                  fields["tgt"].vocab)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)

        wordvec = {'utterance': model_opt.pretrained_wordvec[0]}
        if len(model_opt.pretrained_wordvec) > 1:
            wordvec['kb'] = model_opt.pretrained_wordvec[1]

        def load_wordvec(embeddings, name):
            embeddings.load_pretrained_vectors(
                    wordvec[name], model_opt.fix_pretrained_wordvec)

        # Don't need pretrained word vec for LFs
        if not model_opt.model in ('lf2lf', 'lflm'):
            load_wordvec(model.encoder.embeddings, 'utterance')
            if hasattr(model, 'context_embedder'):
                load_wordvec(model.context_embedder.embeddings, 'utterance')
        # if hasattr(model, 'kb_embedder'):
        #     load_wordvec(model.kb_embedder.embeddings, 'kb')

        if model_opt.model == 'seq2seq':
            load_wordvec(model.decoder.embeddings, 'utterance')

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model
