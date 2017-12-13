from cocoa.systems.rulebased_system import RulebasedSystem as BaseRulebasedSystem, add_rulebased_arguments
from sessions.rulebased_session import RulebasedSession

class RulebasedSystem(BaseRulebasedSystem):

    def __init__(self, lexicon, generator, manager, timed_session, realizer=None):
        super(RulebasedSystem, self).__init__(lexicon, generator, manager, timed_session)
        self.realizer = realizer

    def _new_session(self, agent, kb, config):
        return RulebasedSession(agent, kb, self.lexicon, config, self.generator, self.manager, realizer=self.realizer)
