from cocoa.systems.rulebased_system import RulebasedSystem as BaseRulebasedSystem
from sessions.rulebased_session import RulebasedSession

class RulebasedSystem(BaseRulebasedSystem):

    def __init__(self, tracker, timed_session=False):
        super(RulebasedSystem, self).__init__(timed_session)
        self.tracker = tracker

    def _new_session(self, agent, kb):
        return RulebasedSession.get_session(agent, kb, self.tracker)
