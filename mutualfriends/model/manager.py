from cocoa.model.manager import Manager as BaseManager

class Manager(BaseManager):
    def available_actions(self, state):
        actions = super(Manager, self).available_actions(state)
        masked_actions = ['negative']
        actions = [a for a in actions if not a in masked_actions]
        return actions

    def choose_action(self, state, context=None):
        if state.matched_item:
            return 'select'
        action = super(Manager, self).choose_action(state, context)
        return action
