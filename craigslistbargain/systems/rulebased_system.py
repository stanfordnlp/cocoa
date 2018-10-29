from cocoa.systems.rulebased_system import RulebasedSystem as BaseRulebasedSystem, add_rulebased_arguments
from sessions.rulebased_session import RulebasedSession

class RulebasedSystem(BaseRulebasedSystem):
    def _new_session(self, agent, kb, config=None):
        return RulebasedSession.get_session(agent, kb, self.lexicon, config, self.generator, self.manager)

