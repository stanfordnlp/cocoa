from cocoa.model.manager import Manager as BaseManager

class Manager(BaseManager):
    def available_actions(self, state):
        actions = super(Manager, self).available_actions(state)
        masked_actions = []
        if state.partner_act != 'done':
            masked_actions.append('done')
        actions = [a for a in actions if not a in masked_actions]
        return actions

    def choose_action(self, state, context=None):
        action = super(Manager, self).choose_action(state, context)
        if state.partner_act == 'done':
            return 'done'
        if state.partner_act in ('ask-plot', 'ask'):
            return 'inform'
        if state.partner_act == 'unknown' and state.curr_title is not None:
            return 'inform'
        return action
