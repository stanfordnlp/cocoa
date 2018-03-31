import sys
import os
from itertools import count

from onmt.Utils import use_gpu

from neural.generator import Generator
from neural.utterance import UtteranceBuilder
from neural.beam import Scorer


def add_evaluator_arguments(parser):
    group = parser.add_argument_group('Model')
    group.add_argument('--checkpoint-file', required=True,
          help='Path to model .pt file')

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
    def __init__(self, model, vocab, gt_prefix=1):
        self.model = model
        self.vocab = vocab
        self.gt_prefix = gt_prefix

    def evaluate(self, opt, model_opt, data, split='test'):
        scorer = Scorer(opt.alpha)

        generator = Generator(self.model, self.vocab,
                              beam_size=opt.beam_size,
                              n_best=opt.n_best,
                              max_length=opt.max_length,
                              global_scorer=scorer,
                              cuda=use_gpu(opt),
                              min_length=opt.min_length)

        builder = UtteranceBuilder(self.vocab, opt.n_best, has_tgt=True)

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        out_file = sys.stdout

        data_iter = data.generator(split, shuffle=False)
        num_batches = data_iter.next()
        for batch in data_iter:
            batch_data = generator.generate_batch(batch, gt_prefix=self.gt_prefix)
            utterances = builder.from_batch(batch_data)

            for trans in utterances:
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                gold_score_total += trans.gold_score
                gold_words_total += len(trans.gold_sent)

                n_best_preds = [" ".join(pred)
                                for pred in trans.pred_sents[:opt.n_best]]
                out_file.write('\n'.join(n_best_preds))
                out_file.write('\n')
                out_file.flush()

                if opt.verbose:
                    sent_number = next(counter)
                    output = trans.log(sent_number)
                    os.write(1, output.encode('utf-8'))
