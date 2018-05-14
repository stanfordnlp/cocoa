import torch
from torch.nn import LogSoftmax
from torch.autograd import Variable
from utterance import UtteranceBuilder
from cocoa.neural.generator import Generator, Sampler

class FBnegSampler(Sampler):
    def _run_attention_memory(self, batch, enc_memory_bank):
        context_inputs = batch.context_inputs
        _, context_memory_bank = self.model.context_embedder(context_inputs)
        scene_inputs = batch.scene_inputs
        scene_memory_bank = self.model.kb_embedder(scene_inputs)

        memory_banks = [enc_memory_bank, context_memory_bank, scene_memory_bank]
        return memory_banks

    def generate_batch(self, batch, gt_prefix=1, enc_state=None):
        # (1) Run the encoder on the src.
        lengths = batch.lengths
        dec_states, enc_memory_bank = self._run_encoder(batch, enc_state)
        memory_banks = self._run_attention_memory(batch, enc_memory_bank)

        # (1.1) Go over forced prefix.
        inp = batch.decoder_inputs[:gt_prefix]
        decoder_outputs, dec_states, _ = self.model.decoder(
            inp, memory_banks, dec_states, memory_lengths=lengths)

        # (2) Sampling
        batch_size = batch.size
        preds = []
        for i in xrange(self.max_length):
            # Outputs to probs
            decoder_outputs = decoder_outputs.squeeze(0)  # (batch_size, rnn_size)
            out = self.model.generator.forward(decoder_outputs).data  # Logprob (batch_size, vocab_size)
            # Sample with temperature
            scores = out.div(self.temperature)
            scores.sub_(scores.max(1, keepdim=True)[0].expand(scores.size(0), scores.size(1)))
            pred = torch.multinomial(scores.exp(), 1).squeeze(1)  # (batch_size,)
            preds.append(pred)
            # Forward step
            inp = Variable(pred.view(1, -1))  # (seq_len=1, batch_size)
            decoder_outputs, dec_states, _ = self.model.decoder(
                inp, memory_banks, dec_states, memory_lengths=lengths)

        ''' (3) Go over selection of predicted outcome
        Since we are only doing test evaluation, we no longer need
        to output a selection every timestep, instead just once at the end.

        Decoder_outputs  (seq_len, batch_size, hidden_dim) = (1 x 8 x 256)
        '''
        dec_out = decoder_outputs.transpose(0,1)
        scene_out = memory_banks[-1].transpose(0,1).contiguous().view(batch_size, 1, -1)
        select_h = torch.cat([dec_out, scene_out], 2).transpose(0,1)
        # select_h is (1 x batch_size x (rnn_size+(6 * kb_embed_size)) )
        select_h = self.model.select_encoder.forward(select_h)
        # now select_h is (1 x batch_size x kb_embed_size)
        select_out = [decoder.forward(select_h) for decoder in self.model.select_decoders]
        # select_out is a list of 6 items, where each item is (1 x batch_size x kb_vocab_size)
        # after concat shape is num_items x batch_size x kb_vocab_size
        # taking the argmax at dimension 2 gives (6 x batch_size)
        selections = torch.max(torch.cat(select_out), dim=2)[1].transpose(0, 1)
        # Finally, after transpose we get (batch_size x num_items)
        
        # (4) Wrap up predictions for viewing later
        preds = torch.stack(preds).t()  # (batch_size, seq_len)
        # Insert one dimension (n_best) so that its structure is consistent
        # with beam search generator
        preds = preds.unsqueeze(1)
        # TODO: add actual scores
        ret = {"predictions": preds,
                "selections": selections,
               "scores": [[0]] * batch_size,
               "attention": [None] * batch_size,
               "dec_states": dec_states,
               }

        ret["gold_score"] = [0] * batch_size
        ret["batch"] = batch
        return ret
