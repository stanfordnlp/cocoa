__author__ = 'anushabala'
from cocoa.core.systems.system import System
from cocoa.core.sessions.mutualfriends.heuristic_session import HeuristicSession

def add_heuristic_system_arguments(parser):
    parser.add_argument('--joint-facts', default=False, action='store_true', help='Generate joint attributes, e.g., hiking and philosophy')
    parser.add_argument('--ask', default=False, action='store_true', help='Ask questions, e.g., do you have ...')

class HeuristicSystem(System):
    def __init__(self, joint_facts, ask):
        super(HeuristicSystem, self).__init__()
        # Control difficulty
        self.joint_facts = joint_facts
        self.ask = ask

    @classmethod
    def name(cls):
        return 'heuristic'

    def new_session(self, agent, kb):
        return HeuristicSession(agent, kb, self.joint_facts, self.ask)
