__author__ = 'anushabala'
from session import Session


class NeuralSession(Session):
    def __init__(self, agent, kb, lexicon, model, schema):
        super(NeuralSession, self).__init__(agent)
        # todo do we need lexicon and schema here since they've already been used by NeuralSystem to create the
        # embedding model?
        # Create new Graph instance specific to this agent here using the lexicon and schema


    # todo implement send, receive, etc.
