from src.basic.systems.rulebased_system import BaseRulebasedSystem
from src.basic.sessions.negotiation.rulebased_session import RulebasedSession

class RulebasedSystem(BaseRulebasedSystem):

    def __init__(self, lexicon, timed_session=False):
        super(RulebasedSystem, self).__init__(timed_session)
        self.lexicon = lexicon

    def _new_session(self, agent, kb):
        return RulebasedSession.get_session(agent, kb, self.lexicon)
