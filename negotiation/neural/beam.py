from onmt.translate.Beam import Beam


class Scorer(object):
    """
    Re-ranking score.
    """
    def __init__(self, length_alpha):
        self.alpha = length_alpha

    def score(self, beam, logprobs):
        """
        Additional term add to log probability
        See https://arxiv.org/pdf/1609.08144.pdf.
        """
        l_term = (((5 + len(beam.next_ys)) ** self.alpha) /
                  ((5 + 1) ** self.alpha))
        return (logprobs / l_term)

    def update_global_state(self, beam):
        return

