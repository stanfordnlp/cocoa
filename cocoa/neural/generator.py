import torch
from torch.autograd import Variable

import onmt.io
from onmt.Utils import aeq

from symbols import markers
from beam import Beam
from utterance import UtteranceBuilder


class Generator(object):
    """
    Uses a model to generate a batch of response. Adapted from onmt.translate.Translator.

    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
    """
    def __init__(self, model, vocab,
                 beam_size=1, n_best=1,
                 max_length=100,
                 global_scorer=None, copy_attn=False, cuda=False,
                 beam_trace=False, min_length=0):
        self.model = model
        self.vocab = vocab
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.cuda = cuda
        self.min_length = min_length

        # for debugging
        self.beam_accum = None
        if beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def _run_encoder(self, batch, enc_states=None):
        encoder_inputs = batch.encoder_inputs
        lengths = batch.lengths

        enc_states, memory_bank = self.model.encoder(encoder_inputs, lengths, enc_states)
        dec_states = self.model.decoder.init_decoder_state(
                                        encoder_inputs, memory_bank, enc_states)

        return dec_states, memory_bank

    def _run_attention_memory(self, batch, enc_memory_bank):
        if batch.num_context > 0 and hasattr(self.model, 'kb_embedder'):
            context_inputs = batch.context_inputs
            _, context_memory_bank = self.model.context_embedder(context_inputs)
            memory_bank = [enc_memory_bank, context_memory_bank]

            # TODO: hacky. fix.
            if hasattr(batch, 'title_inputs') and self.model.kb_embedder:
                title_inputs = batch.title_inputs
                _, title_memory_bank = self.model.kb_embedder(title_inputs)
                memory_bank.append(title_memory_bank)

                desc_inputs = batch.desc_inputs
                _, desc_memory_bank = self.model.kb_embedder(desc_inputs)
                memory_bank.append(desc_memory_bank)

            elif hasattr(batch, 'scene_inputs') and self.model.kb_embedder:
                scene_inputs = batch.scene_inputs
                _, scene_memory_bank = self.model.kb_embedder(scene_inputs)
                memory_bank.append(scene_memory_bank)
        else:
            memory_bank = enc_memory_bank
        return memory_bank

    def generate_batch(self, batch, gt_prefix=1, enc_state=None):
        """
        Generate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           gt_prefix (int): ground truth prefix length(bos)

        """

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.size
        vocab = self.vocab

        def get_bos(b):
            """Get the starting symbol. If starting (enforced) prefix is longer
            than 1, use the last symbol as the starting symbol. The rest (previous
            ones) will be force decoded later. See (1.1) Go over forced prefix.
            """
            bos = batch.decoder_inputs[gt_prefix-1][b].data.cpu().numpy()[0]
            return bos

        beam = [Beam(beam_size, n_best=self.n_best,
                     cuda=self.cuda,
                     global_scorer=self.global_scorer,
                     pad=vocab.word_to_ind[markers.PAD],
                     bos=get_bos(b),
                     eos=vocab.word_to_ind[markers.EOS],
                     min_length=self.min_length)
                for b in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a): return Variable(a, volatile=True)

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # (1) Run the encoder on the src.
        lengths = batch.lengths
        dec_states, enc_memory_bank = self._run_encoder(batch, enc_state)
        memory_bank = self._run_attention_memory(batch, enc_memory_bank)

        # (1.1) Go over forced prefix.
        if gt_prefix > 1:
            inp = batch.decoder_inputs[:gt_prefix-1]
            _, dec_states, _ = self.model.decoder(
                inp, memory_bank, dec_states, memory_lengths=lengths)

        # (2) Repeat src objects `beam_size` times.
        #src_map = rvar(batch.src_map.data) \
        #    if data_type == 'text' and self.copy_attn else None
        # TODO: num_context should be a property of the model, not the data!!
        if batch.num_context > 0 and hasattr(self.model, 'context_embedder'):
            memory_bank = [rvar(bank.data) for bank in memory_bank]
        else:
            memory_bank = rvar(memory_bank.data)
        memory_lengths = lengths.repeat(beam_size)
        dec_states.repeat_beam_size_times(beam_size)

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                     .t().contiguous().view(1, -1))

            # Turn any copied words to UNKs
            # 0 is unk
            #if self.copy_attn:
            #    inp = inp.masked_fill(
            #        inp.gt(len(self.vocab) - 1), 0)

            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            #inp = inp.unsqueeze(2)

            # Run one step.
            dec_out, dec_states, attn = self.model.decoder(inp, memory_bank,
                        dec_states, memory_lengths=memory_lengths)
            dec_out = dec_out.squeeze(0)
            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            out = self.model.generator.forward(dec_out).data
            out = unbottle(out)
            # beam x tgt_vocab

            #if not self.copy_attn:
            #    out = self.model.generator.forward(dec_out).data
            #    out = unbottle(out)
            #    # beam x tgt_vocab
            #else:
            #    out = self.model.generator.forward(dec_out,
            #                                       attn["copy"].squeeze(0),
            #                                       src_map)
            #    # beam x (tgt_vocab + extra_vocab)
            #    out = data.collapse_copy_scores(
            #        unbottle(out.data),
            #        batch, self.vocab, data.src_vocabs)
            #    # beam x tgt_vocab
            #    out = out.log()

            # (c) Advance each beam.
            # beam: (batch_size,)
            # out: (beam_size, batch_size, vocab_size)
            for j, b in enumerate(beam):
                b.advance(
                    out[:, j],
                    unbottle(attn["std"]).data[:, j, :memory_lengths[j]])
                dec_states.beam_update(j, b.get_current_origin(), beam_size)

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size
        #if "tgt" in batch.__dict__:
        #    ret["gold_score"] = self._run_target(batch, data)
        ret["batch"] = batch
        return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": [],
               }
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret

    def _run_target(self, batch, data):
        data_type = data.data_type
        if data_type == 'text':
            _, src_lengths = batch.src
        else:
            src_lengths = None
        src = onmt.io.make_features(batch, 'src', data_type)
        tgt_in = onmt.io.make_features(batch, 'tgt')[:-1]

        #  (1) run the encoder on the src
        enc_states, memory_bank = self.model.encoder(src, src_lengths)
        dec_states = \
            self.model.decoder.init_decoder_state(src, memory_bank, enc_states)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        tt = torch.cuda if self.cuda else torch
        gold_scores = tt.FloatTensor(batch.batch_size).fill_(0)
        dec_out, dec_states, attn = self.model.decoder(
            tgt_in, memory_bank, dec_states, memory_lengths=src_lengths)

        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
        for dec, tgt in zip(dec_out, batch.tgt[1:].data):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            gold_scores += scores
        return gold_scores

class Sampler(Generator):
    def __init__(self, model, vocab,
                 temperature=1, max_length=100, cuda=False):
        self.model = model
        self.vocab = vocab
        self.temperature = temperature
        self.max_length = max_length
        self.cuda = cuda
        self.tt = torch.cuda if cuda else torch
        self.eos = vocab.to_ind(markers.EOS)

        # For debugging
        self.builder = UtteranceBuilder(vocab)

    def generate_batch(self, batch, gt_prefix=1, enc_state=None):
        # (1) Run the encoder on the src.
        lengths = batch.lengths
        dec_states, enc_memory_bank = self._run_encoder(batch, enc_state)
        memory_bank = self._run_attention_memory(batch, enc_memory_bank)

        # (1.1) Go over forced prefix.
        inp = batch.decoder_inputs[:gt_prefix]
        dec_out, dec_states, _ = self.model.decoder(
            inp, memory_bank, dec_states, memory_lengths=lengths)

        # (2) Sampling
        batch_size = batch.size
        preds = []
        for i in xrange(self.max_length):
            # Outputs to probs
            dec_out = dec_out.squeeze(0)  # (batch_size, rnn_size)
            out = self.model.generator.forward(dec_out).data  # Logprob (batch_size, vocab_size)
            # Sample with temperature
            scores = out.div(self.temperature)
            scores.sub_(scores.max(1, keepdim=True)[0].expand(scores.size(0), scores.size(1)))
            pred = torch.multinomial(scores.exp(), 1).squeeze(1)  # (batch_size,)
            preds.append(pred)
            # Forward step
            inp = Variable(pred.view(1, -1))  # (seq_len=1, batch_size)
            dec_out, dec_states, _ = self.model.decoder(
                inp, memory_bank, dec_states, memory_lengths=lengths)

        preds = torch.stack(preds).t()  # (batch_size, seq_len)
        # Insert one dimension (n_best) so that its structure is consistent
        # with beam search generator
        preds = preds.unsqueeze(1)
        # TODO: add actual scores
        ret = {"predictions": preds,
               "scores": [[0]] * batch_size,
               "attention": [None] * batch_size,
               "dec_states": dec_states,
               }

        ret["gold_score"] = [0] * batch_size
        ret["batch"] = batch
        return ret

class LMSampler(Sampler):
    def generate_batch(self, batch, gt_prefix=1, enc_state=None):
        # (1.1) Go over forced prefix.
        inp = batch.inputs
        outputs, enc_state = self.model(inp, batch.lengths, enc_state)
        dec_out = outputs[-1:]

        # (2) Sampling
        batch_size = batch.size
        preds = []
        for i in xrange(self.max_length):
            # Outputs to probs
            dec_out = dec_out.squeeze(0)  # (batch_size, rnn_size)
            out = self.model.generator.forward(dec_out).data  # Logprob (batch_size, vocab_size)
            # Sample with temperature
            scores = out.div(self.temperature)
            scores.sub_(scores.max(1, keepdim=True)[0].expand(scores.size(0), scores.size(1)))
            pred = torch.multinomial(scores.exp(), 1).squeeze(1)  # (batch_size,)
            preds.append(pred)
            # Forward step
            inp = Variable(pred.view(1, -1))  # (seq_len=1, batch_size)
            dec_out, enc_state = self.model(inp, None, enc_state=enc_state)

        preds = torch.stack(preds).t()  # (batch_size, seq_len)
        # Insert one dimension (n_best) so that its structure is consistent
        # with beam search generator
        preds = preds.unsqueeze(1)
        # TODO: add actual scores
        ret = {"predictions": preds,
               "scores": [[0]] * batch_size,
               "attention": [None] * batch_size,
               "dec_states": enc_state,
               }

        ret["gold_score"] = [0] * batch_size
        ret["batch"] = batch
        return ret
