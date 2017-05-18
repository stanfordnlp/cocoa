__author__ = 'anushabala'


class FinishedState(object):
    def __init__(self, message, num_seconds, mturk_code=None):
        self.message = message
        self.num_seconds = num_seconds
        self.mturk_code = mturk_code


class WaitingState(object):
    def __init__(self, message, num_seconds):
        if message and len(message) > 0:
            self.message = message
        else:
            self.message = "Please wait while we try to find someone to pair you up with.."
        self.num_seconds = num_seconds


class SurveyState(object):
    def __init__(self, message, agent_idx, scenario_id, kb, partner_kb, attributes, result):
        self.message = message
        self.agent_idx = agent_idx
        self.kb = kb
        self.partner_kb = partner_kb
        self.attributes = attributes
        self.result = result
        self.scenario_id = scenario_id


class UserChatState(object):
    def __init__(self, agent_index, scenario_id, chat_id, kb, attributes, num_seconds, partner_kb=None):
        self.agent_index = agent_index
        self.scenario_id = scenario_id
        self.chat_id = chat_id
        self.kb = kb
        self.attributes = attributes
        self.num_seconds = num_seconds
        self.partner_kb = partner_kb

    def to_dict(self):
        return {"agent_index": self.agent_index,
                "scenario_id": self.scenario_id,
                "chat_id": self.chat_id,
                "kb": self.kb.to_dict(),
                "num_seconds": self.num_seconds,
                "partner_kb": self.partner_kb.to_dict()}