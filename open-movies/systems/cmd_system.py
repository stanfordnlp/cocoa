from cocoa.systems.system import System as BaseSystem
from sessions.cmd_session import CmdSession

class CmdSystem(BaseSystem):
    def __init__(self):
        super(CmdSystem, self).__init__()

    @classmethod
    def name(cls):
        return 'cmd'

    def new_session(self, agent, kb):
        return CmdSession(agent, kb)
