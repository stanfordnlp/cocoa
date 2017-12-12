from cocoa.model.manager import Manager as BaseManager

class Manager(BaseManager):
    def available_actions(self, state):
        actions = super(Manager, self).available_actions(state)
        # These actions should be decided by the bot's strategy
        masked_actions = ['reject', 'accept', 'quit', 'agree']
        if state.num_inquiry > 1:
            masked_actions.append('inquiry')
            if state.curr_price is None:
                actions = ['init-price']
        if state.partner_price is None or state.curr_price is None:
            masked_actions.append('offer')
        actions = [a for a in actions if not a in masked_actions]
        return actions

    def choose_action(self, state, context=None):
        action = super(Manager, self).choose_action(state, context)
        if action == 'offer' and state.partner_act == 'unknown':
            return 'agree'
        return action

