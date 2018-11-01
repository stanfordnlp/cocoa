import sys
import os
from itertools import count
from onmt.Utils import use_gpu

from cocoa.neural.evaluator import add_evaluator_arguments, \
        Evaluator as BaseEvaluator

from neural.generator import get_generator

class Evaluator(BaseEvaluator):
    def evaluate(self, opt, model_opt, data, split='test'):
        text_generator = FBnegSampler(self.model, self.mappings['tgt_vocab'],
            opt.temperature, opt.max_length, use_gpu(opt))

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
            enc_state = dec_state.hidden if dec_state is not None else None

            batch_data = text_generator.generate_batch(batch, model_opt.model,
                        gt_prefix=self.gt_prefix, enc_state=enc_state)
            utterances = self.builder.from_batch(batch_data)
            selections = batch_data["selections"]

            for i, response in enumerate(utterances):
                pred_score_total += response.pred_scores[0]
                pred_words_total += len(response.pred_sents[0])
                gold_score_total += response.gold_score
                gold_words_total += len(response.gold_sent)

            if opt.verbose:
                counter = self.print_results(model_opt, batch, counter,
                        selections, utterances)

    def print_results(self, model_opt, batch, counter, selections, utterances):
        scenes = batch.scene_inputs.transpose(0,1)
        enc_inputs = batch.encoder_inputs.transpose(0,1)
        for i, response in enumerate(utterances):
            sent_number = next(counter)
            scene = self.builder.scene_to_sent(scenes[i], self.mappings['kb_vocab'])
            if selections is not None:
                select = self.builder.selection_to_sent(selections[i],
                                            self.mappings['kb_vocab'])
            else:
                select = [" "]
            print("--------- Example {} -----------".format(sent_number))
            for item_sentence in scene:
                print item_sentence
            # if model_opt.model in ["sum2sum", "sum2seq"]:
            #     summary = self.builder.var_to_sent(enc_inputs[i])
            #     print("SUMMARY: {}".format(summary) )
            output = response.log(sent_number)
            for selection_sentence in select:
                print selection_sentence
            os.write(1, output.encode('utf-8'))

        return counter
