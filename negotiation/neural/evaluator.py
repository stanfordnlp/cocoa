import sys
import os
from itertools import count

from onmt.Utils import use_gpu

from generator import Generator
from utterance import UtteranceBuilder
from beam import Scorer
from symbols import markers


def add_evaluator_arguments(parser):
    group = parser.add_argument_group('Model')
    group.add_argument('--checkpoint-files', nargs='+', required=True,
          help='Path to model .pt file, can be multiple files')

    group = parser.add_argument_group('Beam')
    group.add_argument('--beam-size',  type=int, default=5,
                       help='Beam size')
    group.add_argument('--min-length', type=int, default=1,
                       help='Minimum prediction length')
    group.add_argument('--max-length', type=int, default=50,
                       help='Maximum prediction length.')
    group.add_argument('--n-best', type=int, default=1,
                help="""If verbose is set, will output the n_best decoded sentences""")
    group.add_argument('--alpha', type=float, default=0.5,
                help="""length penalty parameter (higher = longer generation)""")

    group = parser.add_argument_group('Efficiency')
    group.add_argument('--batch-size', type=int, default=30,
                       help='Batch size')
    group.add_argument('--gpuid', default=[], nargs='+', type=int,
                       help="Use CUDA on the listed devices.")

    group = parser.add_argument_group('Logging')
    group.add_argument('--verbose', action="store_true",
                       help='Print scores and predictions for each sentence')



class Evaluator(object):
    def __init__(self, model, mappings, gt_prefix=1):
        self.model = model
        self.gt_prefix = gt_prefix
        self.utterance_vocab = mappings['utterance_vocab']
        self.kb_vocab = mappings['kb_vocab']

    def evaluate(self, opt, model_opt, data, split='test'):
        scorer = Scorer(opt.alpha)

        generator = Generator(self.model, self.utterance_vocab,
                              beam_size=opt.beam_size,
                              n_best=opt.n_best,
                              max_length=opt.max_length,
                              global_scorer=scorer,
                              cuda=use_gpu(opt),
                              min_length=opt.min_length)

        builder = UtteranceBuilder(self.utterance_vocab, opt.n_best, has_tgt=True)

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        data_iter = data.generator(split, shuffle=False)
        num_batches = data_iter.next()
        for batch in data_iter:
            batch_data = generator.generate_batch(batch, gt_prefix=self.gt_prefix)
            utterances = builder.from_batch(batch_data)
            titles = batch.title_inputs.transpose(0,1)
            enc_inputs = batch.encoder_inputs.transpose(0,1)

            for i, response in enumerate(utterances):
                pred_score_total += response.pred_scores[0]
                pred_words_total += len(response.pred_sents[0])
                gold_score_total += response.gold_score
                gold_words_total += len(response.gold_sent)

                if opt.verbose:
                    sent_number = next(counter)
                    title = builder.var_to_sent(titles[i], self.kb_vocab)
                    summary = builder.var_to_sent(enc_inputs[i])
                    print("--------- {0}: {1} -----------".format(sent_number, title))
                    if model_opt.model in ["sum2sum", "sum2seq"]:
                        print("SUMMARY: {}".format(summary) )
                    output = response.log(sent_number)
                    os.write(1, output.encode('utf-8'))
