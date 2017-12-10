from cocoa.model.manager import Manager as BaseManager

class Manager(BaseManager):
    def available_actions(self, state):
        actions = super(Manager, self).available_actions(state)
        masked_actions = []
        if state.curr_proposal is None:
            masked_actions.extend(['select', 'agree'])
        actions = [a for a in actions if not a in masked_actions]
        return actions

    def choose_action(self, state, context=None):
        action = super(Manager, self).choose_action(state, context)
        if action in ('select', 'agree') and not state.my_act in ('clarify', 'agree'):
            return 'clarify'
        if state.my_act == 'clarify':
            if state.partner_act in ('propose', 'insist'):
                if state.partner_proposal and state.partner_proposal != state.my_proposal:
                    return 'propose'
            if state.partner_act == 'disagree':
                return 'propose'
            return 'select'
        return action

###  ================  RESPONSE POLICY  ================= ###
#       Input State                 Output Action           #
#       -----------                 -------------           #
#         greet           -->           greet               #
#      inquire-plot       -->         movie-plot            #
#    inquire-opinion      -->        movie-opinion          #
#     inquire-movie       -->         movie-other           #
#     inquire-title       -->        favorite-title         #
#     inquire-genre       -->        favorite-genre         #
#     inquire-actor       -->        favorite-actor         #
#    inquire-unknown      -->           generic             #
#    favorite-actor       --> favorite-actor, tellmore-actor#
#     favorite-genre      --> favorite-genre, tellmore-genre#
#     favorite-title      -->  movie-opinion, tellmore-title#
#         unknown         -->       inquire, generic        #
### ===================================================== ###