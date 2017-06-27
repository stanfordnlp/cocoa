import json
from util import generate_uuid
from dataset import Example
from threading import Lock
import src.config as config

class BaseController(object):
    """
    Interface of the controller: takes two systems and can run them to generate a dialgoue.
    """
    def __init__(self, scenario, sessions, chat_id=None, debug=True):

        self.lock = Lock()
        self.scenario = scenario
        self.sessions = sessions
        self.chat_id = chat_id
        assert len(self.sessions) == 2
        if debug:
            for agent in (0, 1):
                self.scenario.kbs[agent].dump()
        self.events = []

    def event_callback(self, event):
        raise NotImplementedError

    def get_outcome(self):
        raise NotImplementedError

    def get_result(self, agent_idx):
        return None

    def simulate(self, max_turns=None):
        '''
        Simulate a dialogue.
        '''
        self.events = []
        time = 0
        num_turns = 0
        game_over = False
        while not game_over:
            for agent, session in enumerate(self.sessions):
                event = session.send()
                time += 1
                if not event:
                    continue

                event.time = time
                self.event_callback(event)
                self.events.append(event)

                print 'agent=%s: session=%s, event=%s' % (agent, type(session).__name__, event.to_dict())
                num_turns += 1
                if self.game_over() or (max_turns and num_turns >= max_turns):
                    game_over = True
                    break

                for partner, other_session in enumerate(self.sessions):
                    if agent != partner:
                        other_session.receive(event)

        uuid = generate_uuid('E')
        outcome = self.get_outcome()
        print 'outcome: %s' % outcome
        # TODO: add configurable names to systems and sessions
        return Example(self.scenario, uuid, self.events, outcome, uuid, None)

    def step(self, backend=None):
        '''
        Called by the web backend.
        '''
        with self.lock:
            # try to send messages from one session to the other(s)
            for agent, session in enumerate(self.sessions):
                if session is None:
                    # fail silently, this means that the session has been reset and the controller is effectively
                    # inactive
                    continue
                event = session.send()
                if event is None:
                    continue

                self.event_callback(event)
                self.events.append(event)
                if backend is not None:
                    backend.add_event_to_db(self.get_chat_id(), event)

                for partner, other_session in enumerate(self.sessions):
                    if agent != partner:
                        other_session.receive(event)

    def inactive(self):
        """
        Return whether this controller is currently controlling an active chat session or not (by checking whether both
        users are still active or not)
        :return: True if the chat is active (if both sessions are not None), False otherwise
        """
        for s in self.sessions:
            if s is None:
                return True
        return False

    def set_inactive(self, agents=[]):
        """
        Set any number of sessions in the Controller to None to mark the Controller as inactive. The default behavior
        is to set all sessions to None (if no parameters are supplied to the function), but a list of indices can be
        passed to set the Session objects at those indices to None.
        :param agents: List of indices of Sessions to mark inactive. If this is None, the function is a no-op. If no
        list is passed, the function sets all Session objects to None.
        """
        with self.lock:
            if agents is None:
                return
            elif len(agents) == 0:
                self.sessions = [None] * len(self.sessions)
            else:
                for idx in agents:
                    self.sessions[idx] = None

    def get_chat_id(self):
        return self.chat_id

    def game_over(self):
        raise NotImplementedError

    def complete(self):
        raise NotImplementedError

class Controller(object):
    '''
    Factory of controllers.
    '''
    @staticmethod
    def get_controller(scenario, sessions, chat_id=None, debug=True):
        if config.task == config.MutualFriends:
            return MutualFriendsController(scenario, sessions, chat_id, debug)
        elif config.task == config.Negotiation:
            return NegotiationController(scenario, sessions, chat_id, debug)
        else:
            raise ValueError('Unknown task: %s.' % config.task)

class MutualFriendsController(BaseController):
    def __init__(self, scenario, sessions, chat_id=None, debug=True):
        super(MutualFriendsController, self).__init__(scenario, sessions, chat_id, debug)
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


class NegotiationController(BaseController):
    def __init__(self, scenario, sessions, chat_id=None, debug=True):
        super(NegotiationController, self).__init__(scenario, sessions, chat_id, debug)
        # self.prices = [None, None]
        self.offers = [None, None]
        # self.sides = [None, None]
        self.outcomes = [None, None]
        self.quit = False

    def event_callback(self, event):
        if event.action == 'offer':
            self.offers[event.agent] = event.data
        elif event.action == 'accept':
            self.outcomes[event.agent] = True
        elif event.action == 'reject':
            self.outcomes[event.agent] = False
        elif event.action == 'quit':
            self.quit = True

    def get_outcome(self):
        offer = None
        reward = 0
        if self.offers[0] is not None and self.outcomes[1] is True:
            reward = 1
            offer = self.offers[0]
        elif self.offers[1] is not None and self.outcomes[0] is True:
            reward = 1
            offer = self.offers[1]
        else:
            if (self.offers[0] is not None or self.offers[1] is not None) and False in self.outcomes:
                reward = 0
                offer = self.offers[0] if self.offers[1] is None else self.offers[1]

        # possible outcomes:
        # reward is 1 and offer is not null: complete dialogue
        # reward is 0 and offer is not null: incomplete dialogue (disagreement): offer was made and not accepted
        # reweard is 0 and offer is null: incomplete dialogue: no offer was made
        return {'reward': reward, 'offer': offer}

    def game_over(self):
        return not self.inactive() and \
               ((self.offers[0] is not None and self.outcomes[1] is not None) or
                (self.offers[1] is not None and self.outcomes[0] is not None) or
                 self.quit)

    def get_result(self, agent_idx):
        # todo fix this if we ever want to display results in the survey
        return None

    def complete(self):
        return (self.offers[0] is not None and self.outcomes[1] is True) or (self.offers[1] is not None and self.outcomes[0] is True)

    def get_winner(self):
        # todo fix this if we ever want to calculate who the winner is
        return -1