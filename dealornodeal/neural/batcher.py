import numpy as np
from itertools import izip_longest, izip

import torch
from torch.autograd import Variable

from symbols import markers

def pad_list_to_array(l, fillvalue, dtype):
    '''
    l: list of lists with unequal length
    return: np array with minimal padding
    '''
    return np.array(list(izip_longest(*l, fillvalue=fillvalue)), dtype=dtype).T

class Batch(object):
    def __init__(self, encoder_args, decoder_args, context_data, vocab,
                time_major=True, sort_by_length=True, num_context=None, cuda=False):
        self.vocab = vocab
        self.num_context = num_context
        self.encoder_inputs = encoder_args['inputs']
        self.decoder_inputs = decoder_args['inputs']
        self.context_inputs = encoder_args['context'][0]
        self.scene_inputs = decoder_args['scenarios']  # ie. KB

        self.selections = decoder_args['selections']  # ie. outcome
        self.targets = decoder_args['targets']
        self.size = self.targets.shape[0]
        self.context_data = context_data

        unsorted_attributes = ['encoder_inputs', 'decoder_inputs', 'lengths',
                    'scene_inputs', 'selections', 'targets', 'context_inputs']
        batch_major_attributes = ['encoder_inputs', 'decoder_inputs',
                    'scene_inputs', 'selections', 'targets', 'context_inputs']

        self.lengths, sorted_ids = self.sort_by_length(self.encoder_inputs)
        self.tgt_lengths, _ = self.sort_by_length(self.decoder_inputs)
        if sort_by_length:
            for k, v in self.context_data.iteritems():
                if v is not None:
                    self.context_data[k] = self.order_by_id(v, sorted_ids)
            for attr in unsorted_attributes:
                sorted_attrs = self.order_by_id(getattr(self, attr), sorted_ids)
                setattr(self, attr, sorted_attrs)

        if time_major:
            for attr in batch_major_attributes:
                setattr(self, attr, np.swapaxes(getattr(self, attr), 0, 1))

        # To tensor/variable
        self.encoder_inputs = self.to_variable(self.encoder_inputs, 'long', cuda)
        self.decoder_inputs = self.to_variable(self.decoder_inputs, 'long', cuda)
        self.scene_inputs = self.to_variable(self.scene_inputs, 'long', cuda)
        self.selections = self.to_variable(self.selections, 'long', cuda)
        self.targets = self.to_variable(self.targets, 'long', cuda)
        self.lengths = self.to_tensor(self.lengths, 'long', cuda)
        self.tgt_lengths = self.to_tensor(self.tgt_lengths, 'long', cuda)
        self.context_inputs = self.to_variable(self.context_inputs, 'long', cuda)

    @classmethod
    def to_tensor(cls, data, dtype, cuda=False):
        if type(data) == np.ndarray:
            data = data.tolist()
        if dtype == "long":
            tensor = torch.LongTensor(data)
        elif dtype == "float":
            tensor = torch.FloatTensor(data)
        else:
            raise ValueError
        return tensor.cuda() if cuda else tensor

    @classmethod
    def to_variable(cls, data, dtype, cuda=False):
        tensor = cls.to_tensor(data, dtype)
        var = Variable(tensor)
        return var.cuda() if cuda else var

    def sort_by_length(self, inputs):
        """
        Args:
            inputs (numpy.ndarray): (batch_size, seq_length)
        """
        pad = self.vocab.word_to_ind[markers.PAD]
        def get_length(seq):
            for i, x in enumerate(seq):
                if x == pad:
                    return i
            return len(seq)
        lengths = [get_length(s) for s in inputs]
        # TODO: look into how it works for all-PAD seqs
        lengths = [l if l > 0 else 1 for l in lengths]
        sorted_id = np.argsort(lengths)[::-1]
        return lengths, sorted_id

    def order_by_id(self, inputs, ids):
        if ids is None:
            return inputs
        else:
            if type(inputs) is np.ndarray:
                return inputs[ids, :]
            elif type(inputs) is list:
                return [inputs[i] for i in ids]
            else:
                raise ValueError('Unknown input type {}'.format(type(inputs)))


class DialogueBatcher(object):
    def __init__(self, kb_pad=None, mappings=None, model='seq2seq', num_context=2):
        self.pad = mappings['utterance_vocab'].to_ind(markers.PAD)
        self.kb_pad = kb_pad
        self.mappings = mappings
        self.model = model
        self.num_context = num_context

    def _normalize_dialogue(self, dialogues):
        '''
        All dialogues in a batch should have the same number of turns.
        '''
        max_num_turns = max([d.num_turns for d in dialogues])
        for dialogue in dialogues:
            dialogue.pad_turns(max_num_turns)
        num_turns = dialogues[0].num_turns
        return num_turns

    def _get_turn_batch_at(self, dialogues, STAGE, i):
        pad = self.mappings['utterance_vocab'].to_ind(markers.PAD)
        if i is None:
            # Return all turns
            return [self._get_turn_batch_at(dialogues, STAGE, i) for i in xrange(dialogues[0].num_turns)]
        else:
            turns = [d.turns[STAGE][i] for d in dialogues]
            turn_arr = pad_list_to_array(turns, pad, np.int32)
            return turn_arr

    def _create_turn_batches(self):
        turn_batches = []
        pad = self.mappings['utterance_vocab'].to_ind(markers.PAD)
        for i in xrange(Dialogue.num_stages):
            try:
                for j in xrange(self.num_turns):
                    one_turn = [d.turns[i][j] for d in self.dialogues]
                    turn_batch = pad_list_to_array(one_turn, pad, np.int32)
                    turn_batches.append([turn_batch])
            except IndexError:
                print 'num_turns:', self.num_turns
                for dialogue in self.dialogues:
                    print len(dialogue.turns[0]), len(dialogue.roles)
                import sys; sys.exit()
        return turn_batches

    def pick_helpers(self, token_candidates, candidates):
        helpers = []
        for b, cands in enumerate(token_candidates):
            best_score = 0
            helper_id = -1
            for i, cand in enumerate(cands):
                try:
                    cand = cand['response']
                # Actual number of candidates can be smaller than num_candidates,
                # in which case an empty dict is returned instead
                except KeyError:
                    continue
                if i == 0:
                    target = cand
                    #print 'TARGET:', target
                else:
                    score = compute_bleu(cand, target)
                    #print 'CANDIDATE:', score, cand
                    if score > best_score:
                        best_score = score
                        helper_id = i
            if helper_id == -1:
                helper_id = 0
            #print 'HELPER:', helper_id, cands[helper_id]['response']
            helpers.append(candidates[b][helper_id])
        return np.array(helpers)

    def _remove_last(self, array, value, pad):
        array = np.copy(array)
        nrows, ncols = array.shape
        for i in xrange(nrows):
            for j in xrange(ncols-1, -1, -1):
                if array[i][j] == value:
                    array[i][j] = pad
                    break
        return array

    def _remove_prompt(self, input_arr):
        '''
        Remove starter symbols (e.g. <go>) used for the decoder.
        input_arr: (batch_size, seq_len)
        '''
        # TODO: depending on prompt length
        return input_arr[:, 1:]

    def get_encoder_inputs(self, encoder_turns):
        # Most recent partner utterance
        encoder_inputs = self._remove_prompt(encoder_turns[-1])
        return encoder_inputs

    def get_encoder_context(self, encoder_turns, num_context):
        # |num_context| utterances before the last partner utterance
        encoder_context = [self._remove_prompt(turn) for turn in encoder_turns[-1*(num_context+1):-1]]
        if len(encoder_context) < num_context:
            batch_size = encoder_turns[0].shape[0]
            empty_context = np.full([batch_size, 1], self.pad, np.int32)
            for i in xrange(num_context - len(encoder_context)):
                encoder_context.insert(0, empty_context)
        return encoder_context

    def make_decoder_inputs_and_targets(self, decoder_turns, target_turns=None):
        if target_turns is not None:
            # Decoder inputs: start from <go> to generate, i.e. <go> <token>
            assert decoder_turns.shape == target_turns.shape

        # NOTE: For each row, replace the last <eos> to <pad> to be consistent:
        # 1) The longest sequence does not have </s> as input.
        # 2) At inference time, decoder stops at </s>.
        # 3) Only matters when the model is stateful (decoder state is passed on).
        eos = self.mappings['tgt_vocab'].to_ind(markers.EOS)
        pad = self.mappings['tgt_vocab'].to_ind(markers.PAD)
        decoder_inputs = self._remove_last(decoder_turns, eos, pad)[:, :-1]

        if target_turns is not None:
            decoder_targets = target_turns[:, 1:]
        else:
            decoder_targets = decoder_turns[:, 1:]

        return decoder_inputs, decoder_targets

    def _create_one_batch(self, encoder_turns, decoder_turns, target_turns,
            agents, uuids, kbs, scenarios, selections,
            num_context=None, encoder_tokens=None, decoder_tokens=None):
        encoder_inputs = self.get_encoder_inputs(encoder_turns)
        encoder_context = self.get_encoder_context(encoder_turns, num_context)

        decoder_inputs, decoder_targets = self.make_decoder_inputs_and_targets(decoder_turns, target_turns)

        encoder_args = {
                'inputs': encoder_inputs,
                'context': encoder_context,
                }
        decoder_args = {
                'inputs': decoder_inputs,
                'targets': decoder_targets,
                'scenarios': scenarios,
                'selections': selections,
                }
        context_data = {
                'encoder_tokens': encoder_tokens,
                'decoder_tokens': decoder_tokens,
                'agents': agents,
                'kbs': kbs,
                'uuids': uuids,
                }
        batch = {
                'encoder_args': encoder_args,
                'decoder_args': decoder_args,
                'context_data': context_data,
                }
        return batch

    def int_to_text(self, array, textint_map, stage):
        tokens = [str(x) for x in textint_map.int_to_text((x for x in array if x != self.pad), stage)]
        return ' '.join(tokens)

    def list_to_text(self, tokens):
        return ' '.join(str(x) for x in tokens)

    def print_batch(self, batch, example_id, textint_map, preds=None):
        i = example_id
        print '-------------- Example {} ----------------'.format(example_id)
        if len(batch['decoder_tokens'][i]) == 0:
            print 'PADDING'
            return False
        print 'RAW INPUT:\n {}'.format(self.list_to_text(batch['encoder_tokens'][i]))
        print 'RAW TARGET:\n {}'.format(self.list_to_text(batch['decoder_tokens'][i]))
        print 'ENC INPUT:\n {}'.format(self.int_to_text(batch['encoder_args']['inputs'][i], textint_map, 'encoding'))
        print 'DEC INPUT:\n {}'.format(self.int_to_text(batch['decoder_args']['inputs'][i], textint_map, 'decoding'))
        print 'TARGET:\n {}'.format(self.int_to_text(batch['decoder_args']['targets'][i], textint_map, 'target'))
        if preds is not None:
            print 'PRED:\n {}'.format(self.int_to_text(preds[i], textint_map, 'target'))
        return True

    def _get_token_turns_at(self, dialogues, i):
        stage = 0
        if not hasattr(dialogues[0], 'token_turns'):
            return None
        # Return None for padded turns
        return [dialogue.token_turns[i] if i < len(dialogue.token_turns) else ''
                for dialogue in dialogues]

    def _get_dialogue_data(self, dialogues):
        '''
        Data at the dialogue level, i.e. same for all turns.
        '''
        agents = [d.agents[1] for d in dialogues]  # Decoding agent
        kbs = [d.kb for d in dialogues]
        uuids = [d.uuid for d in dialogues]
        scenarios = [d.scenario for d in dialogues]
        selections = [d.selection for d in dialogues]
        return {
                'agents': agents,
                'kbs': kbs,
                'uuids': uuids,
                'scenarios': scenarios,
                'selections': selections,
                }

    def get_encoding_turn_ids(self, num_turns):
        # NOTE: when creating dialogue turns (see add_utterance), we have set the first utterance to be from the encoding agent
        encode_turn_ids = range(0, num_turns-1, 2)
        return encode_turn_ids

    def _get_lf_batch_at(self, dialogues, i):
        pad = self.mappings['lf_vocab'].to_ind(markers.PAD)
        return pad_list_to_array([d.lfs[i] for d in dialogues], pad, np.int32)

    def create_batch(self, dialogues):
        num_turns = self._normalize_dialogue(dialogues)
        dialogue_data = self._get_dialogue_data(dialogues)

        dialogue_class = type(dialogues[0])
        ENC, DEC, TARGET = dialogue_class.ENC, dialogue_class.DEC, dialogue_class.TARGET

        encode_turn_ids = self.get_encoding_turn_ids(num_turns)
        encoder_turns_all = self._get_turn_batch_at(dialogues, ENC, None)

        # NOTE: encoder_turns contains all previous dialogue context, |num_context|
        # decides how many turns to use
        batch_seq = [
            self._create_one_batch(
                encoder_turns=encoder_turns_all[:i+1],
                decoder_turns=self._get_turn_batch_at(dialogues, DEC, i+1),
                target_turns=self._get_turn_batch_at(dialogues, TARGET, i+1),
                encoder_tokens=self._get_token_turns_at(dialogues, i),
                decoder_tokens=self._get_token_turns_at(dialogues, i+1),
                agents=dialogue_data['agents'],
                uuids=dialogue_data['uuids'],
                kbs=dialogue_data['kbs'],
                scenarios=dialogue_data['scenarios'],
                selections=dialogue_data['selections'],
                num_context=self.num_context,
                )
            for i in encode_turn_ids
            ]

        # bath_seq: A sequence of batches that can be processed in turn where
        # the state of each batch is passed on to the next batch
        return batch_seq


class DialogueBatcherWrapper(object):
    def __init__(self, batcher):
        self.batcher = batcher
        # TODO: fix kb_pad, hacky
        self.kb_pad = batcher.kb_pad

    def create_batch(self, dialogues):
        raise NotImplementedError

    def get_encoder_inputs(self, encoder_turns):
        return self.batcher.get_encoder_inputs(encoder_turns)

    def get_encoder_context(self, encoder_turns, num_context):
        return self.batcher.get_encoder_context(encoder_turns, num_context)

    def list_to_text(self, tokens):
        return self.batcher.list_to_text(tokens)

    def _get_turn_batch_at(self, dialogues, STAGE, i):
        return self.batcher._get_turn_batch_at(dialogues, STAGE, i)


class DialogueBatcherFactory(object):
    @classmethod
    def get_dialogue_batcher(cls, model, **kwargs):
        if model in ('seq2seq', 'lf2lf'):
            batcher = DialogueBatcher(**kwargs)
        else:
            raise ValueError
        return batcher
