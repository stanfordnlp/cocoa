import sys
import os
from itertools import count

from onmt.Utils import use_gpu

from utterance import UtteranceBuilder
from symbols import markers


class Evaluator(object):
    def __init__(self, model, mappings, generator, builder, gt_prefix=1):
        self.model = model
        self.gt_prefix = gt_prefix
        self.mappings = mappings
        self.generator = generator
        self.builder = builder

    def evaluate(self, opt, model_opt, data, split='test'):
        text_generator = self.generator

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        data_iter = data.generator(split, shuffle=False)
        num_batches = data_iter.next()
        dec_state = None
        for batch in data_iter:
            if batch is None:
                dec_state = None
                continue
            elif not self.model.stateful:
                dec_state = None
            # TODO: this is not really stateful!
            enc_state = dec_state.hidden if dec_state is not None else None
            batch_data = text_generator.generate_batch(batch,
                        gt_prefix=self.gt_prefix, enc_state=enc_state)
            utterances = self.builder.from_batch(batch_data)

            for i, response in enumerate(utterances):
                pred_score_total += response.pred_scores[0]
                pred_words_total += len(response.pred_sents[0])
                gold_score_total += response.gold_score
                gold_words_total += len(response.gold_sent)

            if opt.verbose:
                counter = self.print_results(model_opt, batch, counter, utterances)

    def print_results(self, model_opt, batch, utterances):
        for i, response in enumerate(utterances):
            sent_number = next(counter)
            print("--------- {0}: {1} -----------".format(sent_number, title))
            output = response.log(sent_number)
            os.write(1, output.encode('utf-8'))
