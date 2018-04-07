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
from models import MeanEncoder, StdRNNEncoder, StdRNNDecoder, \
              MultiAttnDecoder, NegotiationModel, NMTModel
from onmt.modules import Embeddings, ImageEncoder, CopyGenerator

from cocoa.io.utils import read_pickle
from cocoa.pt_model.util import use_gpu

from preprocess import markers

def add_model_arguments(parser):
    from onmt.modules.SRU import CheckSRU
    group = parser.add_argument_group('Model')
    group.add_argument('--word-vec-size', type=int, default=300,
                       help='Word embedding size for src and tgt.')
    group.add_argument('--share-decoder-embeddings', action='store_true',
                       help="""Use a shared weight matrix for the input and
                       output word  embeddings in the decoder.""")
    group.add_argument('--encoder-type', type=str, default='rnn',
                       choices=['rnn', 'brnn', 'mean', 'transformer', 'cnn'],
                       help="""Type of encoder layer to use. Non-RNN layers
                       are experimental. Options are
                       [rnn|brnn|mean|transformer|cnn].""")
    group.add_argument('--decoder-type', type=str, default='rnn',
                       choices=['rnn', 'transformer', 'cnn'],
                       help="""Type of decoder layer to use. Non-RNN layers
                       are experimental. Options are [rnn|transformer|cnn].""")
    group.add_argument('-copy_attn', action="store_true",
                       help='Train copy attention layer.')
    group.add_argument('--layers', type=int, default=-1,
                       help='Number of layers in enc/dec.')
    group.add_argument('--enc-layers', type=int, default=2,
                       help='Number of layers in the encoder')
    group.add_argument('--dec-layers', type=int, default=2,
                       help='Number of layers in the decoder')
    group.add_argument('--rnn-size', type=int, default=500,
                       help='Size of rnn hidden states')
    group.add_argument('--rnn-type', type=str, default='LSTM',
                       choices=['LSTM', 'GRU', 'SRU'], action=CheckSRU,
                       help="""The gate type to use in the RNNs""")
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
                       help='Model type')
    group.add_argument('--num-context', type=int, default=2,
                       help='Number of sentences to consider as dialogue context (in addition to the encoder input)')

def build_model(model_opt, opt, fields, checkpoint):
    print('Building model...')
    model = onmt.ModelConstructor.make_base_model(model_opt, fields,
                                                  use_gpu(opt), checkpoint)
    if len(opt.gpuid) > 1:
        print('Multi gpu training: ', opt.gpuid)
        model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
    print(model)

    return model

def make_embeddings(opt, word_dict, for_encoder=True):
    """
    Make an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocabulary): words dictionary.
        for_encoder(bool): make Embeddings for encoder or decoder?
    """
    embedding_dim = opt.word_vec_size

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
                          opt.rnn_size, opt.dropout, embeddings,
                          False)


def make_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    bidirectional = True if opt.encoder_type == 'brnn' else False
    if "multibank" in opt.global_attention:
        return MultiAttnDecoder(opt.rnn_type, bidirectional,
                             opt.dec_layers, opt.rnn_size,
                             attn_type=opt.global_attention,
                             dropout=opt.dropout,
                             embeddings=embeddings)
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
                             embeddings=embeddings)

def load_test_model(opt, dummy_opt):
    checkpoint = torch.load(opt.checkpoint_file,
                              map_location=lambda storage, loc: storage)

    model_opt = checkpoint['opt']
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]

    mappings = read_pickle('{}/vocab.pkl'.format(model_opt.mappings))

    model = make_base_model(model_opt, mappings, use_gpu(opt), checkpoint)
    model.eval()
    model.generator.eval()
    return mappings, model, model_opt

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
    src_dict = mappings['vocab']
    src_embeddings = make_embeddings(model_opt, src_dict)
    encoder = make_encoder(model_opt, src_embeddings)
    # Make context embedder.
    if model_opt.num_context > 1:
      original_enc_type = model_opt.encoder_type
      model_opt.encoder_type = "mean"
      context_dict = mappings['vocab']
      context_embeddings = make_embeddings(model_opt, context_dict)
      context_embedder = make_encoder(model_opt, context_embeddings)
      model_opt.encoder_type = original_enc_type
    # Make decoder.
    tgt_dict = mappings['vocab']
    tgt_embeddings = make_embeddings(model_opt, tgt_dict, for_encoder=False)
    decoder = make_decoder(model_opt, tgt_embeddings)

    if "multibank" in model_opt.global_attention:
      model = NegotiationModel(encoder, decoder, context_embedder)
    else:
      print model_opt
      print("should not come here")
      model = NMTModel(encoder, decoder)
    model.model_type = 'text'

    # Make Generator.
    print model_opt.rnn_size
    if not model_opt.copy_attn:
        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, len(tgt_dict)),
            nn.LogSoftmax())
        if model_opt.share_decoder_embeddings:
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
        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                    model_opt.pretrained_wordvec, model_opt.fix_pretrained_wordvec)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                    model_opt.pretrained_wordvec, model_opt.fix_pretrained_wordvec)

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model
