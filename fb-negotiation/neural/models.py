import torch
import torch.nn as nn
import pdb
from cocoa.neural.models import NMTModel
from cocoa.pt_model.util import smart_variable

class NegotiationModel(NMTModel):

    def __init__(self, encoder, decoder, context_embedder, kb_embedder, stateful=False):
        super(NegotiationModel, self).__init__(encoder, decoder, stateful=stateful)
        self.context_embedder = context_embedder
        self.kb_embedder = kb_embedder

    def forward(self, src, tgt, context, scene, lengths, dec_state=None, enc_state=None, tgt_lengths=None):
        enc_final, enc_memory_bank = self.encoder(src, lengths, enc_state)
        _, context_memory_bank = self.context_embedder(context)
        _, scene_memory_bank = self.kb_embedder(scene)
        memory_banks = [enc_memory_bank, context_memory_bank, scene_memory_bank]

        enc_state = self.decoder.init_decoder_state(src, enc_memory_bank, enc_final)
        dec_state = enc_state if dec_state is None else dec_state
        decoder_outputs, dec_state, attns = self.decoder(tgt, memory_banks,
                dec_state, memory_lengths=lengths, lengths=tgt_lengths)

        return decoder_outputs, attns, dec_state

class FBNegotiationModel(NMTModel):

    def __init__(self, encoder, decoder, context_embedder, scene_settings,
            selectors, dropout, model_type, stateful=False):
        super(FBNegotiationModel, self).__init__(encoder, decoder, stateful=stateful)
        self.model_type = model_type
        self.context_embedder = context_embedder
        self.kb_embedder = nn.Embedding(*scene_settings)
        self.dropout = dropout
        self.select_encoder = selectors['enc']   # 1 encoder
        self.select_decoders = selectors['dec']  # 6 decoders

    def forward(self, src, tgt, context, scene, lengths, dec_state=None, enc_state=None, tgt_lengths=None):
        # ---- ENCODING PROCESS -----
        enc_final, enc_memory_bank = self.encoder(src, lengths, enc_state)
        # the memory bas are the RNN hidden states
        context_output, context_memory_bank = self.context_embedder(context)
        scene_memory_bank = self.kb_embedder(scene)
        print("enc: {}".format(enc_memory_bank.shape))
        print("ctx: {}".format(context_memory_bank.shape))
        print("scene: {}".format(scene_memory_bank))
        pdb.set_trace()
        # memory_banks are each (batch_size x seq_len x hidden_size)
        memory_banks = [enc_memory_bank, context_memory_bank, scene_memory_bank]
        # ---- DECODING PROCESS ----
        enc_state = self.decoder.init_decoder_state(src, enc_memory_bank, enc_final)
        dec_state = enc_state if dec_state is None else dec_state
        decoder_outputs, dec_state, attns = self.decoder(tgt, memory_banks,
                dec_state, memory_lengths=lengths, lengths=tgt_lengths)

        # ---- SELECTION PROCESS ----
        if self.model_type == "seq_select":
            dec_seq_len, batch_size, hidden_dim = decoder_outputs.shape
            dec_out = decoder_outputs[-1].unsqueeze(0)
            # enc_final is (2, batch_size, hidden_dim=256)
            # dec_out is (1, batch_size, hidden_dim)
            # context_output is (2, batch_size, hidden_dim)
            # scene_memory_bank is (6, batch_size, kb_embed_size=256))
            sel_hid = torch.cat([enc_final, context_output[0], dec_out, scene_memory_bank], 0)
            # sel_hid is (11, batch_size, (5*hidden_dim + 6*kb_embed_size)
            sel_in = sel_hid.transpose(0,1).contiguous().view(batch_size, 1, -1)
            # sel_in = (16, 1, 11*256) = (16, 1, 2816)
            # no longer have a separate separate encode step, we just go straight to predicting 6 numbers
            sel_out = self.select_decoders.forward(sel_in)
            # sel_out is (16, 1, 60) where vocab size is 10
            selector_outputs = sel_out.view(batch_size, 6, -1).transpose(0,1).contiguous()
            # we end up with (6, 16, vocab_size=10)

            outputs = {
                "decoder": decoder_outputs,
                "selector": selector_outputs,
            }
        else:
            outputs = decoder_outputs

        return outputs, attns, dec_state

        '''
        Old process
        dec_out = decoder_outputs[-1].unsqueeze(0)
        sel_h = torch.cat([enc_final, dec_out, scene_memory_bank], 0)
        # sel_enc_init = smart_variable(torch.zeros(2, batch_size, hidden_dim/2))
        sel_out, sel_h = self.select_encoder.forward(sel_h, context_output[0].contiguous())
        sel_dec_init = sel_out[-1].unsqueeze(0)
        outs = [decoder.forward(sel_dec_init) for decoder in self.select_decoders]
        selector_outputs = torch.cat(outs)

        Old comments
        # select_h is (decoder seq_len + 6) x batch_size x (rnn_size))
        # select_h = self.select_encoder.forward(select_h)
        # select_h = self.dropout(select_h)
        # generate logits for each item separately, outs is a 6-item list
        # outs = [decoder.forward(select_h) for decoder in self.select_decoders]
        '''