from cocoa.systems.rulebased_system import RulebasedSystem as BaseRulebasedSystem
from sessions.hybrid_session import HybridSession

class HybridSystem(BaseRulebasedSystem):

    def _new_session(self, agent, kb, config=None):
        # config should include "use_rl" attribute
        manager_session = self.manager.new_session(agent, kb, config.use_rl)
        return HybridSession.get_session(agent, kb, self.lexicon, config,
                self.generator, manager_session)

    @classmethod
    def name(cls):
        return 'hybrid'

