import torch
from torch.nn import LogSoftmax
from torch.autograd import Variable

from cocoa.core.entity import is_entity

from utterance import UtteranceBuilder
from cocoa.neural.generator import Generator, Sampler as BaseSampler
from symbols import markers, sequence_markers

class Sampler(BaseSampler):
    def _run_attention_memory(self, batch, enc_memory_bank):
        context_inputs = batch.context_inputs
        context_out, context_memory_bank = self.model.context_embedder(context_inputs)
        scene_inputs = batch.scene_inputs
        scene_memory_bank = self.model.kb_embedder(scene_inputs)

        memory_banks = [enc_memory_bank, context_memory_bank, scene_memory_bank]
        #memory_banks = [scene_memory_bank]
        return memory_banks

    def generate_batch(self, batch, gt_prefix=1, enc_state=None, kb=None):
        # This is to ensure we can stop at EOS for stateful models
        assert batch.size == 1

        # (1) Run the encoder on the src.
        lengths = batch.lengths
        encoder_inputs = list(batch.encoder_inputs.data[:, 0])
        #print 'encoder inputs:', map(self.vocab.to_word, encoder_inputs)
        enc_final, enc_memory_bank = self._run_encoder(batch, enc_state)
        memory_banks = self._run_attention_memory(batch, enc_memory_bank)
        #context_out = memory_banks.pop()
        # (1.1) Go over forced prefix.
        inp = batch.decoder_inputs[:gt_prefix]
        #print 'decoder inputs:', map(self.vocab.to_word, inp.data[:, 0])
        decoder_outputs, dec_states, _ = self.model.decoder(
            inp, memory_banks, enc_final, memory_lengths=lengths)

        # (2) Sampling
        batch_size = batch.size
        preds = []
        item_id = 0
        for i in xrange(self.max_length):
            # Outputs to probs
            decoder_outputs = decoder_outputs.squeeze(0)  # (batch_size, rnn_size)
            out = self.model.generator.forward(decoder_outputs).data  # Logprob (batch_size, vocab_size)
            # Sample with temperature
            scores = out.div(self.temperature)

            scores.sub_(scores.max(1, keepdim=True)[0].expand(scores.size(0), scores.size(1)))
            pred = torch.multinomial(scores.exp(), 1).squeeze(1)  # (batch_size,)
            preds.append(pred)
            if pred[0] == self.eos:
                break
            # Forward step
            inp = Variable(pred.view(1, -1))  # (seq_len=1, batch_size)
            decoder_outputs, dec_states, _ = self.model.decoder(
                inp, memory_banks, dec_states, memory_lengths=lengths)

        # (4) Wrap up predictions for viewing later
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

class LFSampler(Sampler):
    def __init__(self, model, vocab,
                 temperature=1, max_length=100, cuda=False):
        super(LFSampler, self).__init__(model, vocab, temperature=temperature, max_length=max_length, cuda=cuda)
        self.eos = self.vocab.to_ind(markers.EOS)
        self.count_actions = map(self.vocab.to_ind, ('insist', 'propose', markers.SELECT))
        counts = [w for w in self.vocab.word_to_ind if '=' in w]
        self.counts = map(self.vocab.to_ind, counts)
        actions = set([w for w in self.vocab.word_to_ind if not
                (w in counts or w in sequence_markers)])
        self.actions = map(self.vocab.to_ind, actions)
        self.select = self.vocab.to_ind(markers.SELECT)

    def get_feasible_counts(self, kb, item_id):
        if not kb:
            return self.counts
        total_count = kb.items[item_id]['Count']
        item = kb.items[item_id]['Name']
        counts = ['{item}={count}'.format(item=item, count=count)
                for count in range(total_count+1)]
        counts = [self.vocab.to_ind(c) for c in counts if self.vocab.has(c)]
        return counts

    def generate_batch(self, batch, gt_prefix=1, enc_state=None, kb=None, select=False):
        # This is to ensure we can stop at EOS for stateful models
        assert batch.size == 1

        # (1) Run the encoder on the src.
        lengths = batch.lengths
        encoder_inputs = list(batch.encoder_inputs.data[:, 0])
        enc_final, enc_memory_bank = self._run_encoder(batch, enc_state)
        memory_banks = self._run_attention_memory(batch, enc_memory_bank)
        # (1.1) Go over forced prefix.
        inp = batch.decoder_inputs[:gt_prefix]
        decoder_outputs, dec_states, _ = self.model.decoder(
            inp, memory_banks, enc_final, memory_lengths=lengths)

        # (2) Sampling
        batch_size = batch.size
        preds = []
        item_id = 0
        for i in xrange(self.max_length):
            # Outputs to probs
            decoder_outputs = decoder_outputs.squeeze(0)  # (batch_size, rnn_size)
            out = self.model.generator.forward(decoder_outputs).data  # Logprob (batch_size, vocab_size)
            # Sample with temperature
            scores = out.div(self.temperature)

            # Masking to ensure valid LF
            # NOTE: batch size must be 1. TODO: relax this restriction
            mask = torch.zeros(scores.size())
            if i > 0:
                if pred[0] in self.count_actions:
                    item_id = 0
                    counts = self.get_feasible_counts(kb, item_id)
                    mask[:, counts] = 1
                elif pred[0] in self.counts:
                    item_id += 1
                    if item_id == len(kb.items):
                        mask[:, self.eos] = 1
                    else:
                        counts = self.get_feasible_counts(kb, item_id)
                        mask[:, counts] = 1
                elif pred[0] in self.actions:
                    mask[:, self.eos] = 1
                else:
                    mask[:, :] = 1
            else:
                if select:
                    mask[:, self.select] = 1
                else:
                    mask[:, self.actions] = 1
            scores[mask == 0] = -1e10

            scores.sub_(scores.max(1, keepdim=True)[0].expand(scores.size(0), scores.size(1)))
            pred = torch.multinomial(scores.exp(), 1).squeeze(1)  # (batch_size,)
            preds.append(pred)
            if pred[0] == self.eos:
                break
            # Forward step
            inp = Variable(pred.view(1, -1))  # (seq_len=1, batch_size)
            decoder_outputs, dec_states, _ = self.model.decoder(
                inp, memory_banks, dec_states, memory_lengths=lengths)

        # (4) Wrap up predictions for viewing later
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



def get_generator(model, vocab, scorer, args, model_args):
    from onmt.Utils import use_gpu
    if args.sample:
        if model_args.model == 'lf2lf':
            generator = LFSampler(model, vocab, args.temperature,
                                max_length=args.max_length,
                                cuda=use_gpu(args))
        else:
            generator = Sampler(model, vocab, args.temperature,
                                max_length=args.max_length,
                                cuda=use_gpu(args))
    else:
        raise ValueError('Beam search not available yet.')
    return generator
