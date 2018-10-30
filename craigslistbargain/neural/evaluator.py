import sys
import os
from itertools import count

from onmt.Utils import use_gpu

from cocoa.neural.evaluator import Evaluator as BaseEvaluator

from neural.generator import get_generator

class Evaluator(BaseEvaluator):
    def print_results(self, model_opt, batch, utterances):
        titles = batch.title_inputs.transpose(0,1)
        enc_inputs = batch.encoder_inputs.transpose(0,1)
        for i, response in enumerate(utterances):
            sent_number = next(counter)
            title = self.builder.var_to_sent(titles[i], self.mappings['kb_vocab'])
            summary = self.builder.var_to_sent(enc_inputs[i])
            print("--------- {0}: {1} -----------".format(sent_number, title))
            if model_opt.model in ["sum2sum", "sum2seq"]:
                print("SUMMARY: {}".format(summary) )
            output = response.log(sent_number)
            os.write(1, output.encode('utf-8'))
