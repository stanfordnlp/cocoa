import sys
import os
from itertools import count

from cocoa.neural.generator import get_generator
from cocoa.neural.evaluator import add_evaluator_arguments, \
        Evaluator as BaseEvaluator


class Evaluator(BaseEvaluator):
    def print_results(self, model_opt, batch, counter, utterances):
        scenes = batch.scene_inputs.transpose(0,1)
        enc_inputs = batch.encoder_inputs.transpose(0,1)
        for i, response in enumerate(utterances):
            sent_number = next(counter)
            scene = self.builder.scene_to_sent(scenes[i], self.mappings['kb_vocab'])
            print("--------- Example {} -----------".format(sent_number))
            for item_sentence in scene:
                print item_sentence
            if model_opt.model in ["sum2sum", "sum2seq"]:
                summary = self.builder.var_to_sent(enc_inputs[i])
                print("SUMMARY: {}".format(summary) )

            output = response.log(sent_number)
            os.write(1, output.encode('utf-8'))

        return counter
