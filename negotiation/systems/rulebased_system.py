from cocoa.systems.rulebased_system import RulebasedSystem as BaseRulebasedSystem
from sessions.rulebased_session import RulebasedSession

def add_rulebased_arguments(parser):
    parser.add_argument('--templates', help='Path to templates (.pkl)')
    parser.add_argument('--rule-model', help='Path to manager model (.pkl)')

class RulebasedSystem(BaseRulebasedSystem):

    def __init__(self, lexicon, timed_session=False, generator=None, manager=None):
        super(RulebasedSystem, self).__init__(timed_session)
        self.lexicon = lexicon
        self.generator = generator
        self.manager = manager

    def _new_session(self, agent, kb, config=None):
        return RulebasedSession.get_session(agent, kb, self.lexicon, config, self.generator, self.manager)

