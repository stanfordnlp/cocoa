from preprocess import markers

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
        output = '\nSENT {}: {}\n'.format(sent_number, self.src_raw)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            # TODO: handle Entity
            tgt_sent = ' '.join([str(x) for x in self.gold_sent])
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += "GOLD SCORE: {:.4f}\n".format(self.gold_score)

        if len(self.pred_sents) > 1:
            output += 'BEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

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

    def _build_target_tokens(self, pred):
        vocab = self.vocab
        tokens = []
        for tok in pred:
            tokens.append(vocab.ind_to_word[tok])
            if tokens[-1] == markers.EOS:
                tokens = tokens[:-1]
                break
        return tokens

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
            src_raw = batch.context_data['encoder_tokens'][b]
            if not batch.context_data['decoder_tokens'][b]:
                continue
            pred_sents = [self._build_target_tokens(preds[b][n])
                          for n in range(self.n_best)]
            gold_sent = None
            if tgt is not None:
                gold_sent = self._build_target_tokens(tgt[:, b])

            utterance = Utterance(src_raw, pred_sents,
                                  attn[b], pred_score[b], gold_sent,
                                  gold_score[b])
            utterances.append(utterance)

        return utterances
