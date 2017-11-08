from cocoa.core.controller import Controller as BaseController

class Controller(BaseController):
    def __init__(self, scenario, sessions, chat_id=None):
        super(Controller, self).__init__(scenario, sessions, chat_id)
        self.quit = False    # happens when "No Deal" button is clicked, cannot be removed
        self.outcomes = [None, None]

    def event_callback(self, event):
        if event.action == 'select':
            self.outcomes[event.agent] = event.data
        elif event.action == 'reject':
            self.quit = True
            self.outcomes[event.agent] = event.data

    def valid_end_state(self):
        try:
            first_agent_proposal = self.outcomes[0]
            second_agent_proposal = self.outcomes[1]
        except TypeError:
            return False

        if (first_agent_proposal is None) or (second_agent_proposal is None):
            # print("first_agent_proposal: {}".format(first_agent_proposal) )
            # print("second_agent_proposal: {}".format(second_agent_proposal) )
            return False

        for item, count in self.scenario.kbs[0].item_counts.iteritems():
            item_proposal = first_agent_proposal[item] + second_agent_proposal[item]
            if int(count) != int(item_proposal):
                return False

        return True

    def calculate_reward(self, agent):
        agent_proposal = self.outcomes[agent]
        total_points = 0
        for item, value in self.scenario.kbs[agent].item_values.iteritems():
            total_points += agent_proposal[item] * value
        return total_points

    def get_result(self, agent):
        """Get result for `agent` so that we can display it on the survey page.
        """
        return self.get_outcome()

    def get_outcome(self):
        if not self.quit and self.valid_end_state():
            first_agent_reward = self.calculate_reward(agent=0)
            second_agent_reward = self.calculate_reward(agent=1)

            reward = {0: first_agent_reward, 1: second_agent_reward}
            valid_deal = True
        else:
            reward = {0: 0, 1: 0}
            valid_deal = False
        split_0 = self.outcomes[0]
        split_1 = self.outcomes[1]

        outcome = {'reward':reward, 'item_split':{0: split_0, 1: split_1}, 'valid_deal':valid_deal, 'agreed': (not self.quit)}
        return outcome

    def game_over(self):
        you_are_still_playing = not self.inactive()
        you_got_points = self.outcomes[0] is not None
        they_got_points = self.outcomes[1] is not None

        if self.quit:
            return True
        elif you_are_still_playing and you_got_points and they_got_points:
            return True
        else:
            return False

    def complete(self):
        """Whether the task was completed successfully, i.e. whether they were able to reach a valid deal.
        """
        if not self.quit and self.valid_end_state():
            return True
        return False
