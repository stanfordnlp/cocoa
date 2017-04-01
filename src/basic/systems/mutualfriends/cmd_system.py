from system import System
from src.basic.sessions.cmd_session import CmdSession

class CmdSystem(System):
    def __init__(self):
        super(CmdSystem, self).__init__()

    @classmethod
    def name(cls):
        return 'cmd'

    def new_session(self, agent, kb):
        return CmdSession(agent, kb)
