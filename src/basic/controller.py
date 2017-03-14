import json
from util import generate_uuid
from dataset import Example
from threading import Lock


class Controller(object):
    """
    The controller takes two systems and can run them to generate a dialgoue.
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
        self.selections = [None, None]
        self.reward = 0
        self.events = []

    def simulate(self, max_turns=100):
        '''Simulate the dialogue.'''
        self.events = []
        time = 0
        self.selections = [None, None]
        self.reward = 0
        num_turns = 0
        timeup = False
        while True:
            for agent, session in enumerate(self.sessions):
                event = session.send()
                time += 1
                if not event:
                    continue
                event.time = time
                self.events.append(event)

                if event.action == 'select':
                    self.selections[agent] = event.data

                print 'agent=%s: session=%s, event=%s' % (agent, type(session).__name__, event.to_dict())
                num_turns += 1
                if num_turns >= max_turns:
                    timeup = True
                for partner, other_session in enumerate(self.sessions):
                    if agent != partner:
                        other_session.receive(event)

                # Game is over when the two selections are the same
                if self.game_over():
                    self.reward = 1
                    break
            if self.game_over() or timeup:
                break

        uuid = generate_uuid('E')
        outcome = {'reward': self.reward}
        print 'outcome: %s' % outcome
        return Example(self.scenario, uuid, self.events, outcome, uuid, None)

    def step(self, backend=None):
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
                self.events.append(event)

                if backend is not None:
                    backend.add_event_to_db(self.get_chat_id(), event)
                if event.action == 'select':
                    self.selections[agent] = event.data
                    if self.game_over():
                        self.reward = 1
                for partner, other_session in enumerate(self.sessions):
                    if agent != partner:
                        other_session.receive(event)

    def game_over(self):
        return not self.inactive() and self.selections[0] is not None and self.selections[0] == self.selections[1]

    def get_outcome(self):
        return {'reward': self.reward}

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
