import sys
import os
from itertools import count

from onmt.Utils import use_gpu

from generator import get_generator
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

    group = parser.add_argument_group('Beam')
    group.add_argument('--sample', action="store_true",
                       help='Sample instead of beam search')
    group.add_argument('--temperature', type=float, default=1,
                help="""Sample temperature""")

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
        self.mappings = mappings

    def evaluate(self, opt, model_opt, data, split='test'):
        scorer = Scorer(opt.alpha)
        generator = get_generator(self.model, self.mappings['tgt_vocab'], scorer, opt)
        builder = UtteranceBuilder(self.mappings['tgt_vocab'], opt.n_best, has_tgt=True)

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
            batch_data = generator.generate_batch(batch, gt_prefix=self.gt_prefix, enc_state=enc_state)
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
                    title = builder.var_to_sent(titles[i], self.mappings['kb_vocab'])
                    summary = builder.var_to_sent(enc_inputs[i])
                    print("--------- {0}: {1} -----------".format(sent_number, title))
                    if model_opt.model in ["sum2sum", "sum2seq"]:
                        print("SUMMARY: {}".format(summary) )
                    output = response.log(sent_number)
                    os.write(1, output.encode('utf-8'))
