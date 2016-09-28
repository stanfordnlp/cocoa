__author__ = 'anushabala'
from system import System
from src.basic.sessions.neural_session import NeuralSession


class NeuralSystem(System):
    def __init__(self, lexicon, model_path, schema):
        super(NeuralSystem, self).__init__()
        # build embedding model here somehow? this should not be in Graph anymore
        # load model from model_path and store in self.model
        self.model = None
        self.schema = schema
        self.lexicon = lexicon

    def new_session(self, agent, kb):
        NeuralSession(agent, kb, self.lexicon, self.model, self.schema)