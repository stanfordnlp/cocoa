__author__ = 'anushabala'
from system import System


class NeuralSystem(System):
    """
    A subclass of the System class that loads a neural model from a specified path and sends and receives message.
    """
    def __init__(self, agent, kb, lexicon, model_path, schema):
        # todo this might also need some way to interface with the lexicon?
        super(NeuralSystem, self).__init__(agent)

    def receive(self, event):
        # encode received message
        raise NotImplementedError

    def send(self):
        # decode 
        raise NotImplementedError

    def reset(self):
        # reset hidden states of model (to prepare for a new conversation)
        raise NotImplementedError
