from cocoa.basic.systems.rulebased_system import BaseRulebasedSystem
from cocoa.basic.sessions.mutualfriends.rulebased_session import RulebasedSession

class RulebasedSystem(BaseRulebasedSystem):

    def __init__(self, lexicon, timed_session=False, consecutive_entity=True, realizer=None):
        super(RulebasedSystem, self).__init__(timed_session)
        self.lexicon = lexicon
        self.consecutive_entity = consecutive_entity
        self.realizer = realizer

    def _new_session(self, agent, kb):
        return RulebasedSession(agent, kb, self.lexicon, self.realizer, self.consecutive_entity)
