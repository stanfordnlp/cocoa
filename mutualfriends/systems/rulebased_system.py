from cocoa.systems.rulebased_system import RulebasedSystem as BaseRulebasedSystem
from sessions.rulebased_session import RulebasedSession

class RulebasedSystem(BaseRulebasedSystem):

    def __init__(self, lexicon, timed_session=False, realizer=None):
        super(RulebasedSystem, self).__init__(timed_session)
        self.lexicon = lexicon
        self.realizer = realizer

    def _new_session(self, agent, kb, config):
        return RulebasedSession(agent, kb, self.lexicon, self.realizer)
