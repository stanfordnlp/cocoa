from cocoa.systems.system import System
from sessions.rl_session import RLSession
from onmt import Optim

class RLSystem(System):
    def __init__(self, system, args):
        self.system = system
        self.optim = self.build_optimizer(args)
        # self.env = env          # should include model attr
        # self.model = env.model  # should include discount attr

    @classmethod
    def name(cls):
        return 'RL-{}'.format(self.system.name())

    def build_optimzer(self, args):
        print('Making optimizer for training.')
        optim = Optim(args.optim, args.learning_rate, args.max_grad_norm,
            model_size=args.rnn_size)
        return optim

    def new_session(self, agent, kb):
        session = self.system.new_session(agent, kb, rl=True)
        self.optim.set_parameters(self.session.model.parameters())
        rl_session = RLSession(session, self.optim)
        return rl_session

