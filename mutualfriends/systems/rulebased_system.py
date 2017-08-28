from cocoa.systems.rulebased_system import RulebasedSystem as BaseRulebasedSystem
from mutualfriends.sessions.rulebased_session import RulebasedSession

class RulebasedSystem(BaseRulebasedSystem):

    def __init__(self, lexicon, timed_session=False, consecutive_entity=True, realizer=None):
        super(RulebasedSystem, self).__init__(timed_session)
        self.lexicon = lexicon
        self.consecutive_entity = consecutive_entity
        self.realizer = realizer

    def _new_session(self, agent, kb):
        return RulebasedSession(agent, kb, self.lexicon, self.realizer, self.consecutive_entity)
