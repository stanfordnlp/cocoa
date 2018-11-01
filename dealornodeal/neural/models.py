import torch
import torch.nn as nn
from cocoa.neural.models import NMTModel

class NegotiationModel(NMTModel):

    def __init__(self, encoder, decoder, context_embedder, selectors, scene_settings,
            dropout, stateful=False):
        super(NegotiationModel, self).__init__(encoder, decoder, stateful=stateful)
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

        # memory_banks are each (seq_len x batch_size x hidden_size)
        memory_banks = [enc_memory_bank, context_memory_bank, scene_memory_bank]

        # ---- DECODING PROCESS ----
        enc_state = self.decoder.init_decoder_state(src, enc_memory_bank, enc_final)
        dec_state = enc_state if dec_state is None else dec_state
        outputs, dec_state, attns = self.decoder(tgt, memory_banks,
                dec_state, memory_lengths=lengths, lengths=tgt_lengths)

        return outputs, attns, dec_state
