from cocoa.core.controller import Controller as BaseController

class Controller(BaseController):
    def __init__(self, scenario, sessions, chat_id=None, debug=False):
        super(Controller, self).__init__(scenario, sessions, chat_id, debug)
        self.marked_agree = [False, False]
        self.quit = False
        self.outcomes = [None, None]

    def event_callback(self, event):
        if event.action == 'select':
            self.marked_agree[event.agent] = True
            self.outcomes[event.agent] = event.data
        elif event.action == 'reject':
            self.quit = True
            self.outcomes[event.agent] = event.data
        elif event.action == 'quit':
            self.quit = True

    def valid_end_state(self):
        try:
            first_agent_proposal = self.outcomes[0]['item_split']
            second_agent_proposal = self.outcomes[1]['item_split']
        except TypeError:
            self.outcomes[0] = {'item_split': 'deal_rejected'}
            self.outcomes[1] = {'item_split': 'deal_rejected'}
            return False
        items = ['book', 'hat', 'ball']

        for item in items:
            item_count = self.scenario.kbs[0].facts['Item_counts']
            item_proposal = first_agent_proposal[item] + second_agent_proposal[item]
            valid_deal = (item_count == item_proposal)
            if not valid_deal:
                return False

        return True

    def postgame_check(self):
        if self.valid_end_state():
            print("Example game ended successfully with a deal.")
        # elif num_turns >= self.max_turns:
        #     print("No deal was made.")
        #     # paper says you get no points when there is no agreement
        #     self.outcomes[0] = {'deal_points': 0, 'item_split': 'no_deal'}
        #     self.outcomes[1] = {'deal_points': 0, 'item_split': 'no_deal'}
        else:
            # print("Incompatiable proposals were made by the two agents.")
            print("Invalid end state or max turns.")
            self.outcomes[0]['deal_points'] = 0
            self.outcomes[1]['deal_points'] = 0

    def get_outcome(self):
        agent_0_reward = self.outcomes[0]['deal_points']
        agent_1_reward = self.outcomes[1]['deal_points']
        split_0 = self.outcomes[0]['item_split']
        split_1 = self.outcomes[1]['item_split']

        return {'reward': agent_0_reward + agent_1_reward, 'item_split_0': split_0, 'item_split_1': split_1}

    def game_over(self):
        you_are_still_playing = not self.inactive()
        you_agreed_and_got_points = (self.marked_agree[0] == True) and (self.outcomes[0] is not None)
        they_agreed_and_got_points = (self.marked_agree[1] == True) and (self.outcomes[1] is not None)

        if you_are_still_playing and you_agreed_and_got_points and they_agreed_and_got_points:
            self.postgame_check()
            return True
        elif you_are_still_playing and self.quit:
            self.postgame_check()
            return True
        else:
            return False