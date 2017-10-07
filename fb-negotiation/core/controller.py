from cocoa.core.controller import Controller as BaseController

class Controller(BaseController):
    def __init__(self, scenario, sessions, chat_id=None):
        super(Controller, self).__init__(scenario, sessions, chat_id)
        #self.marked_agree = [False, False]
        self.quit = False
        self.outcomes = [None, None]

    def event_callback(self, event):
        if event.action == 'select':
            #self.marked_agree[event.agent] = True
            self.outcomes[event.agent] = event.data
        # TODO: remove reject and quit
        elif event.action == 'reject':
            self.quit = True
            self.outcomes[event.agent] = event.data
            self.outcomes[event.agent]['deal'] = 'reject'
        elif event.action == 'quit':
            self.quit = True

    def valid_end_state(self):
        try:
            first_agent_proposal = self.outcomes[0]
            second_agent_proposal = self.outcomes[1]
        except TypeError:
            return False

        #print("first_agent_proposal: {}".format(first_agent_proposal) )
        #print("second_agent_proposal: {}".format(second_agent_proposal) )

        for item, count in self.scenario.kbs[0].item_counts.iteritems():
            item_proposal = first_agent_proposal[item] + second_agent_proposal[item]
            if int(count) != int(item_proposal):
                return False

        return True

    #def postgame_check(self):
    #    if self.valid_end_state():
    #        print("Example game ended successfully with a deal.")
    #        self.outcomes[0]['deal'] = 'success'
    #        self.outcomes[1]['deal'] = 'success'
    #    # elif num_turns >= self.max_turns:
    #    #     print("No deal was made.")
    #    #     # paper says you get no points when there is no agreement
    #    #     self.outcomes[0] = {'deal_points': 0, 'item_split': 'no_deal'}
    #    #     self.outcomes[1] = {'deal_points': 0, 'item_split': 'no_deal'}
    #    else:
    #        # print("Incompatiable proposals were made by the two agents.")
    #        print("Invalid end state or max turns.")
    #        self.outcomes[0]['deal'] = 'fail'
    #        self.outcomes[1]['deal'] = 'fail'
    #        self.outcomes[0]['points'] = 0
    #        self.outcomes[1]['points'] = 0

    def calculate_reward(self, agent):
        agent_proposal = self.outcomes[agent]
        total_points = 0
        for item, value in self.scenario.kbs[agent].item_values.iteritems():
            total_points += agent_proposal[item] * value
        return total_points

    def get_outcome(self):
        if self.valid_end_state():
            reward = {0: self.calculate_reward(agent=0), 1: self.calculate_reward(agent=1)}
        else:
            reward = 0
        split_0 = self.outcomes[0]
        split_1 = self.outcomes[1]
        return {'reward': reward, 'item_split': {0: split_0, 1: split_1}}

    def game_over(self):
        return not self.inactive() and \
                self.outcomes[0] is not None and \
                self.outcomes[1] is not None

        #you_are_still_playing = not self.inactive()
        #you_agreed_and_got_points = (self.marked_agree[0] == True) and (self.outcomes[0] is not None)
        #they_agreed_and_got_points = (self.marked_agree[1] == True) and (self.outcomes[1] is not None)

        #if you_are_still_playing and you_agreed_and_got_points and they_agreed_and_got_points:
        #    self.postgame_check()
        #    return True
        #elif you_are_still_playing and self.quit:
        #    self.postgame_check()
        #    return True
        #else:
        #    return False

    def complete(self):
        return self.outcomes[0] is not None and self.outcomes[1] is not None
        #option1 = (self.marked_agree[1] is True) and (self.outcomes[0] is not None)
        #option2 = (self.marked_agree[0] is True) and (self.outcomes[1] is not None)
        #return (option1 or option2)
