import torch
from torch.nn import LogSoftmax
from torch.autograd import Variable
from utterance import UtteranceBuilder
from cocoa.neural.generator import Generator, Sampler

class FBnegSampler(Sampler):
    def generate_batch(self, batch, gt_prefix=1, enc_state=None):
        # (1) Run the encoder on the src.
        lengths = batch.lengths
        dec_states, enc_memory_bank = self._run_encoder(batch, enc_state)
        memory_bank = self._run_attention_memory(batch, enc_memory_bank)
        print("memory_lengths: {} should be 4".format(len(memory_bank)))
        scene_output = memory_bank.pop()
        print("memory_lengths: {} should now be 3".format(len(memory_bank)))
        # (1.1) Go over forced prefix.
        inp = batch.decoder_inputs[:gt_prefix]
        decoder_outputs, dec_states, _ = self.model.decoder(
            inp, memory_bank, dec_states, memory_lengths=lengths)

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
                inp, memory_bank, dec_states, memory_lengths=lengths)

        ''' (3) Go over selection of predicted outcome
        Since we are only doing test evaluation, we no longer need
        to output a selection every timestep, instead just once at the end.

        Decoder_outputs  (seq_len, batch_size, hidden_dim) = (1 x 4 x 256)
        '''
        print("decoder_outputs: {}".format(decoder_outputs.shape))
        select_h = torch.cat([decoder_outputs, scene_output], 2)
        print("top select_h: {}".format(select_h.shape))
        # select_h is (1 x 4 x (rnn_size + kb_embed size))
        select_h = self.model.select_encoder.forward(select_h)
        print("select_h: {}".format(select_h.shape))
        # generate logits for each item separately, outs is a 6-item list
        select_out = [decoder.forward(select_h) for decoder in self.select_decoders]
        foo = torch.cat(select_out)
        print("concat Outputs: {}".format(foo.shape))
        # after concat shape is 6 x 4 x 28, so now we make prediction
        selections = torch.cat(select_out).argmax(dim=2) # .transpose()
        # now selections is 6 x 4   (num_items, batch_size)

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