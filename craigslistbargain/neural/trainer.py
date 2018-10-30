from cocoa.neural.trainer import Trainer as BaseTrainer

class Trainer(BaseTrainer):
    ''' Class that controls the training process which inherits from Cocoa '''

    def _run_batch(self, batch, dec_state=None, enc_state=None):
        encoder_inputs = batch.encoder_inputs
        decoder_inputs = batch.decoder_inputs
        targets = batch.targets
        lengths = batch.lengths
        #tgt_lengths = batch.tgt_lengths

        # running forward() method in the NegotiationModel
        if hasattr(self.model, 'context_embedder'):
            context_inputs = batch.context_inputs
            title_inputs = batch.title_inputs
            desc_inputs = batch.desc_inputs

            outputs, attns, dec_state = self.model(encoder_inputs,
                    decoder_inputs, context_inputs, title_inputs,
                    desc_inputs, lengths, dec_state, enc_state)
        # running forward() method in NMT Model
        else:
            outputs, attns, dec_state = self.model(encoder_inputs,
                  decoder_inputs, lengths, dec_state, enc_state)

        return outputs, attns, dec_state
