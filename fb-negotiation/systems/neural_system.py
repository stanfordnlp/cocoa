from collections import namedtuple

from cocoa.systems.system import System
from cocoa.sessions.timed_session import TimedSessionWrapper
from sessions.neural_session import NeuralSession

from fb_model import utils
from fb_model.agent import LstmRolloutAgent

def add_neural_system_arguments(parser):
    parser.add_argument('--checkpoint', type=str,
        help='model file')
    parser.add_argument('--temperature', type=float, default=1.0,
        help='temperature')
    parser.add_argument('--gpu', action='store_true',
        help='Use GPU or not')
    #parser.add_argument('--num_types', type=int, default=3,
    #    help='number of object types')
    #parser.add_argument('--num_objects', type=int, default=6,
    #    help='total number of objects')
    #parser.add_argument('--max_score', type=int, default=10,
    #    help='max score per object')
    #parser.add_argument('--score_threshold', type=int, default=6,
    #    help='successful dialog should have more than score_threshold in score')

# `args` for LstmRolloutAgent
Args = namedtuple('Args', ['temperature', 'domain'])

class NeuralSystem(System):
    def __init__(self, model_file, temperature, timed_session=False, gpu=False):
        super(NeuralSystem, self).__init__()
        self.timed_session = timed_session
        self.model = utils.load_model(model_file, gpu=gpu)
        self.args = Args(temperature=temperature, domain='object_division')

    @classmethod
    def name(cls):
        return 'neural'

    def new_session(self, agent, kb):
        session = NeuralSession(agent, kb, self.model, self.args)
        if self.timed_session:
            session = TimedSessionWrapper(session)
        return session

