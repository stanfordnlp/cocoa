from preprocess import markers

class Evaluator(BaseEvaluator):
    def _stop_symbol(self):
        return self.vocab.to_ind(markers.EOS)

    def _remove_symbols(self):
        return map(self.vocab.to_ind, (markers.PAD,))

