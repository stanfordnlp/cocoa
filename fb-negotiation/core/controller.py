from cocoa.core.controller import Controller as BaseController

class Controller(BaseController):
    def __init__(self, scenario, sessions, chat_id=None, debug=False):
        super(Controller, self).__init__(scenario, sessions, chat_id, debug)
        self.marked_agree = [False, False]
        self.quit = False
        self.outcomes = [None, None]
        # self.prices = [None, None]

    def event_callback(self, event):
        if event.action == 'accept':
            self.marked_agree[event.agent] = True
            self.outcomes[event.agent] = event.data
        elif event.action == 'reject':
            self.marked_agree[event.agent] = True
            self.quit = True
            self.outcomes[0] = {'deal_points': 0, 'item_split': 'no_deal'}  # paper says you get no points when there is no agreement
            self.outcomes[1] = {'deal_points': 0, 'item_split': 'no_deal'}  # paper says you get no points when there is no agreement
        elif event.action == 'quit':
            self.quit = True

    def get_outcome(self):
        agent_0_reward = self.outcomes[0]['deal_points']
        agent_1_reward = self.outcomes[1]['deal_points']
        split_0 = self.outcomes[0]['item_split']
        split_1 = self.outcomes[1]['item_split']

        return {'reward': agent_0_reward + agent_1_reward, 'item_split_0': split_0, 'item_split_1': split_1}
        # offer = None
        # reward = 0
        # if self.offers[0] is not None and self.outcomes[1] is True:
        #     reward = 1
        #     offer = self.offers[0]
        # elif self.offers[1] is not None and self.outcomes[0] is True:
        #     reward = 1
        #     offer = self.offers[1]
        # else:
        #     if (self.offers[0] is not None or self.offers[1] is not None) and False in self.outcomes:
        #         reward = 0
        #         offer = self.offers[0] if self.offers[1] is None else self.offers[1]

        # possible outcomes:
        # reward is 1 and offer is not null: complete dialogue
        # reward is 0 and offer is not null: incomplete dialogue (disagreement): offer was made and not accepted
        # reweard is 0 and offer is null: incomplete dialogue: no offer was made
        return {'reward': reward, 'offer': offer}

    def game_over(self):
        you_are_still_playing = not self.inactive()
        you_agreed_and_got_points = (self.marked_agree[0] == True) and (self.outcomes[0] is not None)
        they_agreed_and_got_points = (self.marked_agree[1] == True) and (self.outcomes[1] is not None)

        if you_are_still_playing and you_agreed_and_got_points and they_agreed_and_got_points:
            return True
        elif you_are_still_playing and self.quit:
            return True
        else:
            return False

    def get_result(self, agent_idx):
        # todo fix this if we ever want to display results in the survey
        return None

    def complete(self):
        return (self.offers[0] is not None and self.outcomes[1] is True) or (self.offers[1] is not None and self.outcomes[0] is True)

    def get_winner(self):
        # todo fix this if we ever want to calculate who the winner is
        return -1
