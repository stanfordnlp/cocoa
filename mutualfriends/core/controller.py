from cocoa.core.controller import Controller as BaseController

class Controller(BaseController):
    def __init__(self, scenario, sessions, chat_id=None):
        super(Controller, self).__init__(scenario, sessions, chat_id, allow_cross_talk=True)
        self.selections = [None, None]

    def event_callback(self, event):
        if event.action == 'select':
            self.selections[event.agent] = event.data

    def get_outcome(self):
        if self.selections[0] is not None and self.selections[0] == self.selections[1]:
            reward = 1
        else:
            reward = 0
        return {'reward': reward}

    def game_over(self):
        return not self.inactive() and self.selections[0] is not None and self.selections[0] == self.selections[1]

    def complete(self):
        return self.selections[0] is not None and self.selections[0] == self.selections[1]
