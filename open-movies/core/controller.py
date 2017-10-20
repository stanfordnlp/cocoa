from cocoa.core.controller import Controller as BaseController

class Controller(BaseController):
    def __init__(self, scenario, sessions, chat_id=None):
        super(Controller, self).__init__(scenario, sessions, chat_id)
        self.done = False
        self.human_turns = 0
        self.bot_turns = 0

    def event_callback(self, event):
        if event.action == 'done':
            self.done = True
            # self.outcomes[event.agent] = event.data

    def get_result(self, agent):
        return self.get_outcome()

    def get_outcome(self):
        return {'reward':1, 'done': True} if self.complete() else {'reward':0, 'done': False}

    def game_over(self):
        return True if self.done else False

    def complete(self):
        """Whether the task was completed successfully, i.e. whether they were able to reach a valid deal.
        """
        return True if (self.human_turns >= 5) and (self.bot_turns >= 5) else False
