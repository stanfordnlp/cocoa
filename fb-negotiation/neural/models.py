import torch.nn as nn

from cocoa.neural.models import NMTModel

class LM(nn.Module):
    def __init__(self, encoder):
        super(LM, self).__init__()
        self.encoder = encoder
        self.stateful = False

    def forward(self, src, lengths, enc_state=None):
        enc_final, outputs = self.encoder(src, lengths, enc_state)
        return outputs, enc_final


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

    def __init__(self, encoder, decoder, context_embedder, kb_embedder,
            selectors, dropout, stateful=False):
        super(NegotiationModel, self).__init__(encoder, decoder, stateful=stateful)
        self.context_embedder = context_embedder
        self.kb_embedder = kb_embedder
        self.dropout = dropout
        self.select_encoder = selectors['enc']   # 1 encoder
        self.select_decoders = selectors['dec']  # 6 decoders

    def forward(self, src, tgt, context, scene, lengths, dec_state=None, enc_state=None, tgt_lengths=None):
        # ---- ENCODING PROCESS -----
        enc_final, enc_memory_bank = self.encoder(src, lengths, enc_state)
        # the memory banks are the RNN hidden states
        context_output, context_memory_bank = self.context_embedder(context)
        scene_output, scene_memory_bank = self.kb_embedder(scene)
        memory_banks = [enc_memory_bank, context_memory_bank, scene_memory_bank]

        # ---- DECODING PROCESS ----
        enc_state = self.decoder.init_decoder_state(src, enc_memory_bank, enc_final)
        dec_state = enc_state if dec_state is None else dec_state
        decoder_outputs, dec_state, attns = self.decoder(tgt, memory_banks,
                dec_state, memory_lengths=lengths, lengths=tgt_lengths)

        # ---- SELECTION PROCESS ----
        # concatenate decoder final state and output of the context embedder
        # then resize to the selector hidden state size using select_encoder
        print("dec: {}".format(dec_state.shape))
        print("scene: {}".format(scene_output.shape))
        import pdb; pdb.set_trace()

        select_h = torch.cat([dec_state, scene_output], 2).squeeze(0)
        select_h = self.dropout(select_h)
        select_h = self.select_encoder.forward(select_h)
        # generate logits for each item separately, outs is a 6-item list
        outs = [decoder.forward(select_h) for decoder in self.select_decoders]
        selector_outputs = torch.cat(outs)

        outputs = {
            "decoder": decoder_outputs,
            "selector": selector_outputs
        }

        return outputs, attns, dec_state, selector_outputs

        '''
        Note: FB model performs these alternate steps for selection
             1) concats kb scenario with decoder hidden state
             2) processes further using a separate selector GRU
             3) performs attention over the inputs from part 1
             4) concats the context hidden state and attention results
             5) Pass the final result to the selector encoder
        '''
