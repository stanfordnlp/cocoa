from session import Session

from fb_model import utils
from fb_model.agent import LstmRolloutAgent

class NeuralSession(Session):
    """A wrapper for LstmRolloutAgent.
    """
    def __init__(self, agent, kb, model, args):
        super(NeuralSession, self).__init__(agent)
        self.kb = kb
        self.model = LstmRolloutAgent(model, args)
        context = self.kb_to_context(self.kb)
        self.model.feed_context(context)
        self.state = {
                'selected': False,
                'rejected': False,
                }

    def kb_to_context(self, kb):
        """Convert `kb` to context used by the LstmRolloutAgent.

        Returns:
            context (list[str]):
                [book_count, book_value, hat_count, hat_value, ball_count, ball_value]

        """
        context = []
        for item in ('book', 'hat', 'ball'):
            context.extend([str(kb.item_counts[item]), str(kb.item_values[item])])
        return context

    def parse_choice(self, choice):
        """Convert agent choice to dict.

        Args:
            choice (list[str]):
                e.g. ['item0=1', 'item1=2', 'item2=0', 'item0=0', 'item1=0', 'item2=2'],
                where item 0-2 are book, hat, ball.
                The agent's choice are the first 3 elements.

        Return:
            data (dict): {item: count}

        """
        try:
            return {item: int(choice[i].split('=')[1]) for i, item in enumerate(('book', 'hat', 'ball'))}
        except:
            print 'Cannot parse choice:', choice
            return {item: 0 for i, item in enumerate(('book', 'hat', 'ball'))}

    def receive(self, event):
        if event.action == 'select':
            self.state['selected'] = True
        elif event.action == 'reject':
            self.state['rejected'] = True
        elif event.action == 'message':
            tokens = event.data.lower().strip().split() + ['<eos>']
            self.model.read(tokens)

    def select(self):
        choice = self.model.choose()
        return super(NeuralSession, self).select(self.parse_choice(choice))

    def _is_selection(self, out):
        return len(out) == 1 and out[0] == '<selection>'

    def send(self):
        if self.state['selected']:
            return self.select()

        if self.state['rejected']:
            return self.reject()

        tokens = self.model.write()
        if self._is_selection(tokens):
            return self.select()
        # Omit the last <eos> symbol
        return self.message(' '.join(tokens[:-1]))
