from symbols import markers
from cocoa.core.entity import is_entity

class Utterance(object):
    """
    Contain data of a response prediction.
    """
    def __init__(self, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation to stdout.
        """
        user_utterance = ' '.join([str(x) if is_entity(x) else x for x in self.src_raw])
        output = u'RAW INPUT: {}\n'.format(user_utterance)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join([str(x) for x in best_pred])
        output += 'PRED OUTPUT: {}\n'.format(pred_sent)
        # output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            tgt_sent = ' '.join([str(x) for x in self.gold_sent])
            output += u'GOLD: {}\n'.format(tgt_sent)
            # gold score is always 0 because that is the highest possible
            # output += "GOLD SCORE: {:.4f}\n".format(self.gold_score)

        if len(self.pred_sents) > 1:
            output += 'BEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        output += "\n"

        return output

class UtteranceBuilder(object):
    """
    Build a word-based utterance from the batch output
    of generator and the underlying dictionaries.
    """
    def __init__(self, vocab, n_best=1, has_tgt=False):
        self.vocab = vocab
        self.n_best = n_best
        self.has_tgt = has_tgt
        self.pred_lengths = []

    def build_target_tokens(self, predictions, kb=None):
        tokens = []
        for pred in predictions:
            token = self.vocab.to_word(pred)
            if token == markers.EOS:
                break
            tokens.append(token)
        return tokens

    def entity_to_str(self, entity_tokens, kb):
        return [self._entity_to_str(token, kb) if is_entity(token) else token
                for token in entity_tokens]

    def var_to_sent(self, variables, vocab=None):
        if not vocab:
            vocab = self.vocab

        sent_ids = variables.data.cpu().numpy()
        pad_id = vocab.to_ind(markers.PAD)
        sent_words = [vocab.to_word(x) for x in sent_ids if x != pad_id]
        sent_strings = [str(x) if is_entity(x) else x for x in sent_words]
        readable_sent = ' '.join(sent_strings)

        return readable_sent

    def _entity_to_str(self, entity_token, kb):
        raise NotImplementedError

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert(len(translation_batch["gold_score"]) ==
               len(translation_batch["predictions"]))
        batch_size = batch.size

        # We don't need to sort as batcher already sorted everything

        preds, pred_score, attn, gold_score = (
                        translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["attention"],
                        translation_batch["gold_score"],
                        )

        tgt = batch.targets.data

        utterances = []
        for b in range(batch_size):
            #src_raw = batch.context_data['encoder_tokens'][b]
            src_raw = map(self.vocab.to_word, batch.encoder_inputs.data[:, b])
            if not batch.context_data['decoder_tokens'][b]:
                continue
            pred_sents = [self.build_target_tokens(preds[b][n])
                          for n in range(self.n_best)]
            gold_sent = None
            if tgt is not None:
                #gold_sent = self.build_target_tokens(tgt[:, b])
                gold_sent = map(self.vocab.to_word, tgt[:, b])

            utterance = Utterance(src_raw, pred_sents,
                                  attn[b], pred_score[b], gold_sent,
                                  gold_score[b])
            utterances.append(utterance)

        return utterances

    def calculate_lengths(self, preds):
        total_len = len(preds)
        # TODO: this doesn't work with Marker class
        #marker_len = len([x for x in preds if x in markers])
        entity_len = len([x for x in preds if is_entity(x)])
        keyword_len = total_len - marker_len - entity_len
        return (total_len, keyword_len, marker_len, entity_len)
