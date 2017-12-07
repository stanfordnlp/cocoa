import numpy as np
from itertools import izip_longest, izip

def pad_list_to_array(l, fillvalue, dtype):
    '''
    l: list of lists with unequal length
    return: np array with minimal padding
    '''
    return np.array(list(izip_longest(*l, fillvalue=fillvalue)), dtype=dtype).T

class DialogueBatcher(object):
    def __init__(self, int_markers=None, slot_filling=None, kb_pad=None):
        self.slot_filling = slot_filling
        self.int_markers = int_markers
        self.kb_pad = kb_pad

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
        if i is None:
            # Return all turns
            return [self._get_turn_batch_at(dialogues, STAGE, i) for i in xrange(dialogues[0].num_turns)]
        else:
            turns = [d.turns[STAGE][i] for d in dialogues]
            turn_arr = pad_list_to_array(turns, self.int_markers.PAD, np.int32)
            return turn_arr

    def _create_turn_batches(self):
        turn_batches = []
        #for i in xrange(Dialogue.num_stages):
        for i in xrange(3):
            try:
                turn_batches.append([pad_list_to_array(
                    [d.turns[i][j] for d in self.dialogues], self.int_markers.PAD, np.int32)
                    for j in xrange(self.num_turns)])
            except IndexError:
                print 'num_turns:', self.num_turns
                for dialogue in self.dialogues:
                    print len(dialogue.turns[0]), len(dialogue.roles)
                import sys; sys.exit()
        return turn_batches

    def _create_price_batches(self):
        price_batches = [pad_list_to_array(
            [dialogue.price_turns[j] for dialogue in self.dialogues], self.int_markers.PAD, np.float32)
            for j in xrange(self.num_turns)]
        return price_batches

    def create_context_batch(self, dialogues, pad):
        category_batch = np.array([d.category for d in dialogues], dtype=np.int32)
        # TODO: make sure pad is consistent
        #pad = Dialogue.mappings['kb_vocab'].to_ind(markers.PAD)
        title_batch = pad_list_to_array([d.title for d in dialogues], pad, np.int32)
        description_batch = pad_list_to_array([d.description for d in dialogues], pad, np.int32)
        return {
                'category': category_batch,
                'title': title_batch,
                'description': description_batch,
                }

    def _get_agent_batch_at(self, dialogues, i):
        return [dialogue.agents[i] for dialogue in dialogues]

    def _get_kb_batch(self, dialogues):
        return [dialogue.kb for dialogue in dialogues]

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

    def _mask_slots(self, targets):
        '''
        targets: target sequence (interger)
        return: mask (set to 1) words between <slot> and </slot> (including </slot> but not <slot>)
        '''
        mask = np.zeros_like(targets)
        for i, target in enumerate(targets):
            delimiters = np.where((target == self.int_markers.START_SLOT) | (target == self.int_markers.END_SLOT))[0]
            assert len(delimiters) % 2 == 0
            for j in xrange(0, delimiters.shape[0], 2):
                start, end = delimiters[j], delimiters[j+1]
                # Include </slot> but not <slot>
                mask[i][start+1:end+1] = 1
        targets[mask == 0] = self.int_markers.PAD
        return targets

    def _remove_last(self, array, value):
        nrows, ncols = array.shape
        for i in xrange(nrows):
            for j in xrange(ncols-1, -1, -1):
                if array[i][j] == value:
                    array[i][j] = self.int_markers.PAD
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
            empty_context = np.full([batch_size, 1], self.int_markers.PAD, np.int32)
            for i in xrange(num_context - len(encoder_context)):
                encoder_context.insert(0, empty_context)
        return encoder_context

    def _create_one_batch(self, encoder_turns=None, decoder_turns=None, target_turns=None, agents=None, uuids=None, kbs=None, kb_context=None, num_context=None, encoder_tokens=None, decoder_tokens=None):
        encoder_inputs = self.get_encoder_inputs(encoder_turns)
        encoder_context = self.get_encoder_context(encoder_turns, num_context)

        # Decoder inputs: start from <go> to generate, i.e. <go> <token>
        assert decoder_turns.shape == target_turns.shape
        decoder_inputs = np.copy(decoder_turns)
        # NOTE: For each row, replace the last <eos> to <pad> to be consistent:
        # 1) The longest sequence does not have </s> as input.
        # 2) At inference time, decoder stops at </s>.
        # 3) Only matters when the model is stateful (decoder state is passed on).
        decoder_inputs = self._remove_last(decoder_inputs, self.int_markers.EOS)[:, :-1]
        decoder_targets = target_turns[:, 1:]

        # Mask for slots
        #if self.slot_filling:
        #    decoder_targets = self._mask_slots(decoder_targets)

        #helpers = self.pick_helpers(token_candidates, candidates)  # (batch_size, helper_len)
        #context_batch['helper'] = helpers

        encoder_args = {
                'inputs': encoder_inputs,
                'context': encoder_context,
                }
        decoder_args = {
                'inputs': decoder_inputs,
                'targets': decoder_targets,
                'context': kb_context,
                }
        batch = {
                'encoder_args': encoder_args,
                'decoder_args': decoder_args,
                'encoder_tokens': encoder_tokens,
                'decoder_tokens': decoder_tokens,
                'agents': agents,
                'kbs': kbs,
                'uuids': uuids,
                'size': len(agents),
                }
        return batch

    def int_to_text(self, array, textint_map, stage):
        tokens = [str(x) for x in textint_map.int_to_text((x for x in array if x != self.int_markers.PAD), stage)]
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
        agents = self._get_agent_batch_at(dialogues, 1)  # Decoding agent
        kbs = self._get_kb_batch(dialogues)
        uuids = [d.uuid for d in dialogues]
        kb_context_batch = self.create_context_batch(dialogues, self.kb_pad)
        return {
                'agents': agents,
                'kbs': kbs,
                'uuids': uuids,
                'kb_context': kb_context_batch,
                }

    def get_encoding_turn_ids(self, num_turns):
        # NOTE: when creating dialogue turns (see add_utterance), we have set the first utterance to be from the encoding agent
        encode_turn_ids = range(0, num_turns-1, 2)
        return encode_turn_ids

    def create_batch(self, dialogues):
        num_turns = self._normalize_dialogue(dialogues)
        dialogue_data = self._get_dialogue_data(dialogues)

        dialogue_class = type(dialogues[0])
        ENC, DEC, TARGET = dialogue_class.ENC, dialogue_class.DEC, dialogue_class.TARGET
        num_context = dialogue_class.num_context

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
                kb_context=dialogue_data['kb_context'],
                num_context=num_context,
                )
            for i in encode_turn_ids
            ]

        # bath_seq: A sequence of batches that can be processed in turn where
        # the state of each batch is passed on to the next batch
        batch = {
                 'batch_seq': batch_seq,
                }

        return batch

class DialogueBatcherWrapper(object):
    def __init__(self, batcher):
        self.batcher = batcher
        # TODO: fix kb_pad, hacky
        self.kb_pad = batcher.kb_pad

    def create_batch(self, dialogues):
        raise NotImplementedError

    def create_context_batch(self, dialogues, pad):
        return self.batcher.create_context_batch(dialogues, pad)

    def get_encoder_inputs(self, encoder_turns):
        return self.batcher.get_encoder_inputs(encoder_turns)

    def get_encoder_context(self, encoder_turns, num_context):
        return self.batcher.get_encoder_context(encoder_turns, num_context)

    def list_to_text(self, tokens):
        return self.batcher.list_to_text(tokens)

    def _get_turn_batch_at(self, dialogues, STAGE, i):
        return self.batcher._get_turn_batch_at(dialogues, STAGE, i)

class PriceWrapper(DialogueBatcherWrapper):
    '''
    Add prices in the input. Used for PricePredictor.
    '''
    def _get_price_batch_at(self, dialogues, i):
        prices = [d.price_turns[i] for dialogue in dialogues]
        price_arr = pad_list_to_array(prices, self.int_markers.PAD, np.float32)
        return price_arr

    def _create_one_batch(self, encoder_prices=None, decoder_prices=None):
        # Remove pad (<go>) at the beginning of utterance
        encoder_price_inputs = encoder_prices[:, 1:]
        decoder_price_inputs = decoder_prices[:, :-1]
        price_targets = decoder_prices[:, 1:]
        return {
                'encoder_price_inputs': encoder_price_inputs,
                'decoder_price_inputs': decoder_price_inputs,
                'price_targets': price_targets,
                }

    def create_batch(self, dialogues):
        dialogue_batch = self.batcher.create_batch(dialogues)

        num_turns = dialogues[0].num_turns
        encode_turn_ids = self.batcher.get_encoding_turn_ids(num_turns)
        price_batch_seq = [
                self.create_one_batch(
                    encoder_prices=self._get_price_batch_at(dialogues, i),
                    decoder_prices=self._get_price_batch_at(dialogues, i+1),
                    )
                for i in encode_turn_ids
                ]

        for batch, price_batch in izip(dialogue_batch['batch_seq'], price_batch_seq):
            batch['encoder_args']['price_inputs'] = price_batch['encoder_price_inputs']
            batch['decoder_args']['price_inputs'] = price_batch['decoder_price_inputs']
            batch['decoder_args']['price_targets'] = price_batch['price_targets']

        return dialogue_batch


class RetrievalWrapper(DialogueBatcherWrapper):
    '''
    Add response candidates from the retriever.
    '''
    def _get_token_candidates_at(self, dialogues, i):
        return [dialogue.token_candidates[i] if i < len(dialogue.token_candidates) else ''
                for dialogue in dialogues]

    def _max_num_candidates(self, dialogues, i):
        return max([len(d.candidates[i]) for d in dialogues])

    def _get_candidate_batch_at(self, dialogues, i):
        candidates = [d.candidates[i] for d in dialogues]
        max_num_candidates = self._max_num_candidates(dialogues, i)
        # Padding
        candidates = [c + [[] for _ in xrange(max_num_candidates - len(c))] for c in candidates]  # (batch_size, num_candidates, seq_len)
        # Flatten
        candidates = [c for cands in candidates for c in cands]

        candidate_arr = pad_list_to_array(candidates, self.batcher.int_markers.PAD, np.int32)
        batch_size = len(dialogues)
        candidate_arr = candidate_arr.reshape(batch_size, max_num_candidates, -1)
        return candidate_arr

    def _get_label_batch_at(self, dialogues, i):
        batch_size = len(dialogues)
        num_candidates = self._max_num_candidates(dialogues, i)
        labels = np.zeros((batch_size, num_candidates), dtype=np.int32)
        if dialogues[0].true_candidate_inds is None:
            return labels
        true_candidate_inds = [d.true_candidate_inds[i] if i < len(d.true_candidate_inds) else [] for d in dialogues]
        batch_inds = [[b]*len(candidate_inds) for b, candidate_inds in enumerate(true_candidate_inds)]
        flatten = lambda l: [x for ll in l for x in ll]
        labels[flatten(batch_inds), flatten(true_candidate_inds)] = 1
        return labels

    def _create_one_batch(self, candidates=None, candidate_labels=None, token_candidates=None):
        if candidate_labels is not None:
            assert candidates.shape[:2] == candidate_labels.shape
        return {
                'candidates': candidates,
                'token_candidates': token_candidates,
                'candidate_labels': candidate_labels,
                }

    def create_batch(self, dialogues):
        dialogue_batch = self.batcher.create_batch(dialogues)

        num_turns = dialogues[0].num_turns
        encode_turn_ids = self.batcher.get_encoding_turn_ids(num_turns)
        candidate_batch_seq = [
                self._create_one_batch(
                    candidates=self._get_candidate_batch_at(dialogues, i+1),
                    candidate_labels=self._get_label_batch_at(dialogues, i+1),
                    token_candidates=self._get_token_candidates_at(dialogues, i+1),
                    )
                for i in encode_turn_ids
                ]

        for batch, candidate_batch in izip(dialogue_batch['batch_seq'], candidate_batch_seq):
            batch['decoder_args']['candidates'] = candidate_batch['candidates']
            batch['decoder_args']['candidate_labels'] = candidate_batch['candidate_labels']
            batch['token_candidates'] = candidate_batch['token_candidates']

        return dialogue_batch

    def print_batch(self, batch, example_id, textint_map, preds=None):
        success = self.batcher.print_batch(batch, example_id, textint_map)
        if success:
            candidates = batch['decoder_args']['candidates'][example_id]
            if preds is not None:
                ranks = preds[example_id]
            else:
                ranks = [-1] * len(candidates)
            for i, (cand, rank) in enumerate(izip(candidates, ranks)):
                print 'CANDIDATE {} (label={})'.format(i, rank)
                if cand[0] == self.batcher.int_markers.PAD:
                    print 'PADDING'
                else:
                    print self.batcher.int_to_text(cand, textint_map, 'decoding')
            return True
        else:
            return False

class LMDialogueBatcher(DialogueBatcher):
    def _create_one_batch_lm(self, tokens):
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        batch = {
                'inputs': inputs,
                'targets': targets,
                }
        return batch

    def _create_one_batch(self, encode_turn, decode_turn, target_turn, price_encode_turn, price_decode_turn, encode_tokens, decode_tokens, token_candidates, candidates, agents, kbs, context_batch):
        encoder_inputs = encode_turn

        # Remove pad (<go>) at the beginning of utterance
        #encoder_price_inputs = price_encode_turn[:, 1:]

        # Decoder inputs: start from <go> to generate, i.e. <go> <token>
        assert decode_turn.shape == target_turn.shape
        decoder_inputs = np.copy(decode_turn)

        # Include pad (<go>) at the beginning of utterance
        #decoder_price_inputs = price_decode_turn[:, :-1]
        #price_targets = price_decode_turn[:, 1:]

        decoder_targets = target_turn[:, 1:]

        #helpers = self.pick_helpers(token_candidates, candidates)  # (batch_size, helper_len)
        #context_batch['helper'] = helpers

        # TODO: group these
        batch = {
                 'encoder_inputs': encoder_inputs,
                 'decoder_inputs': decoder_inputs,
                 'targets': decoder_targets,
                 'encoder_tokens': encode_tokens,
                 'decoder_tokens': decode_tokens,
                 'token_candidates': token_candidates,
                 'candidates': candidates,
                 'agents': agents,
                 'kbs': kbs,
                 'context': context_batch,
                }
        return batch

    def create_batch(self, bptt_steps):
        lm_batch = self._create_lm_batch(bptt_steps)
        batch = super(LMDialogueBatcher, self).create_batch()
        context = batch['batch_seq'][0]['context']
        batch['eval_batch_seq'] = batch['batch_seq']
        for b in lm_batch:
            b['context'] = context
        batch['batch_seq'] = lm_batch
        return batch

    def _create_lm_batch(self, bptt_steps=35):
        data = [d.join_turns() for d in self.dialogues]
        dialogue_tokens = [d[0] for d in data]
        # TODO: don't need dialogue_turns
        dialogue_turns = [d[1] for d in data]
        max_len = max([len(tokens) for tokens in dialogue_tokens])
        batch_size = len(self.dialogues)
        T = np.full([batch_size, max_len], self.int_markers.PAD, dtype=np.int32)
        for i, tokens in enumerate(dialogue_tokens):
            T[i, :len(tokens)] = tokens

        batch_seq = []
        for i in range(0, max_len, bptt_steps):
            if i + bptt_steps > max_len - 5:
                batch_seq.append(self._create_one_batch_lm(T[:, i:]))
                break
            else:
                batch_seq.append(self._create_one_batch_lm(T[:, i:i+bptt_steps]))

        return batch_seq


class EvalDialogueBatcher(DialogueBatcher):
    def _create_one_batch(self, encode_turn, decode_turn, target_turn, price_encode_turn, price_decode_turn, encode_tokens, decode_tokens, token_candidates, candidates, agents, kbs, context_batch, num_context, candidate_scores):
        batch = super(EvalDialogueBatcher, self)._create_one_batch(encode_turn, decode_turn, target_turn, price_encode_turn, price_decode_turn, encode_tokens, decode_tokens, token_candidates, candidates, agents, kbs, context_batch, num_context)
        batch['candidate_scores'] = candidate_scores
        return batch

    def _create_candidate_batches(self):
        candidate_batches = [self._normalize_candidates(
            [dialogue.candidates[0] for dialogue in self.dialogues])]
        return candidate_batches

    def create_batch(self):
        self._normalize_dialogue()
        turn_batches = self._create_turn_batches()  # (batch_size, num_turns)
        price_batches = self._create_price_batches()  # (batch_size, num_turns, price_feat_size)
        candidate_batches = self._create_candidate_batches()  # (batch_size, num_candidate, seq_len)

        enc, dec, tgt = Dialogue.ENC, Dialogue.DEC, Dialogue.TARGET

        # Add agents and kbs
        agents = self._get_agent_batch_at(1)  # Decoding agent
        kbs = self._get_kb_batch()

        pad = EvalDialogue.mappings['kb_vocab'].to_ind(markers.PAD)
        context_batch = self.create_context_batch(pad)

        # We just need one batch (context, response, candidates, scores)
        # TODO: better way to handle Dialogue/EvalDialogue static variables
        # TODO: add candidates batch
        batch = [self._create_one_batch(
                turn_batches[enc][:-1], turn_batches[dec][-1], turn_batches[tgt][-1],
                None, None,
                self._get_context(), self._get_response(),
                self._get_token_candidates_at(), candidate_batches[0],
                agents, kbs, context_batch,
                EvalDialogue.num_context,
                self._get_candidate_scores())]
        return {'batch_seq': batch}

    def _get_context(self):
        return [dialogue.token_turns[:-1] for dialogue in self.dialogues]

    def _get_response(self):
        return [dialogue.token_turns[-1] for dialogue in self.dialogues]

    def _get_token_candidates_at(self):
        return [dialogue.token_candidates[0] for dialogue in self.dialogues]

    def _get_candidate_scores(self):
        return [dialogue.candidate_scores for dialogue in self.dialogues]

class DialogueBatcherFactory(object):
    @classmethod
    def get_dialogue_batcher(cls, model_config, **kwargs):
        batcher = DialogueBatcher(**kwargs)
        if 'price' in model_config:
            batcher = PriceWrapper(batcher)
        if 'retrieve' in model_config:
            batcher = RetrievalWrapper(batcher)
        return batcher
