from cocoa.model.manager import Manager as BaseManager

class Manager(BaseManager):
    def available_actions(self, state):
        actions = super(Manager, self).available_actions(state)
        masked_actions = []
        if state.num_inquiry > 1:
            masked_actions.append('inquiry')
            if state.curr_price is None:
                actions = ['init-price']
        if state.partner_price is None or state.curr_price is None or 'offer' in state.done:
            masked_actions.append('offer')
        actions = [a for a in actions if not a in masked_actions]
        return actions


