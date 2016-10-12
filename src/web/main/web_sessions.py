__author__ = 'anushabala'


class FinishedSession(object):
    def __init__(self, message, num_seconds, mturk_code=None):
        self.message = message
        self.num_seconds = num_seconds
        self.mturk_code = mturk_code


class WaitingSession(object):
    def __init__(self, message, num_seconds):
        if message and len(message) > 0:
            self.message = message
        else:
            self.message = "Please wait while we try to find someone to pair you up with.."
        self.num_seconds = num_seconds


class UserChatSession(object):
    def __init__(self, room_id, agent_index, kb, num_seconds):
        self.room_id = room_id
        self.agent_index = agent_index
        self.kb = kb
        self.num_seconds = num_seconds

    def to_dict(self):
        return {"room": self.room_id,
                "agent_index": self.agent_index,
                "kb": self.kb,
                "num_seconds": self.num_seconds}


class SurveySession(object):
    def __init__(self):
        pass