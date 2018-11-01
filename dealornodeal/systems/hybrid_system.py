from cocoa.systems.rulebased_system import RulebasedSystem as BaseRulebasedSystem
from sessions.hybrid_session import HybridSession

class HybridSystem(BaseRulebasedSystem):

    def _new_session(self, agent, kb, use_rl=False):
        self.manager.timed_session = False
        manager_session = self.manager.new_session(agent, kb)
        return HybridSession(agent, kb, self.lexicon, None,
                self.generator, manager_session)

    @classmethod
    def name(cls):
        return 'hybrid'

