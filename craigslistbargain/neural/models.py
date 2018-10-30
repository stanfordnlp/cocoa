import torch.nn as nn

from cocoa.neural.models import NMTModel

class NegotiationModel(NMTModel):

    def __init__(self, encoder, decoder, context_embedder, kb_embedder, stateful=False):
        super(NegotiationModel, self).__init__(encoder, decoder, stateful=stateful)
        self.context_embedder = context_embedder
        self.kb_embedder = kb_embedder

    def forward(self, src, tgt, context, title, desc, lengths, dec_state=None, enc_state=None, tgt_lengths=None):
        enc_final, enc_memory_bank = self.encoder(src, lengths, enc_state)
        _, context_memory_bank = self.context_embedder(context)
        if self.kb_embedder:
            _, title_memory_bank = self.kb_embedder(title)
            _, desc_memory_bank = self.kb_embedder(desc)
            memory_banks = [enc_memory_bank, context_memory_bank, title_memory_bank, desc_memory_bank]
        else:
            memory_banks = [enc_memory_bank, context_memory_bank]

        enc_state = self.decoder.init_decoder_state(src, enc_memory_bank, enc_final)
        dec_state = enc_state if dec_state is None else dec_state
        decoder_outputs, dec_state, attns = self.decoder(tgt, memory_banks,
                dec_state, memory_lengths=lengths, lengths=tgt_lengths)

        return decoder_outputs, attns, dec_state
