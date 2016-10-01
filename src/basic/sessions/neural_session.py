__author__ = 'anushabala'
from session import Session


class NeuralSession(Session):
    def __init__(self, agent, kb, lexicon, vocab, tf_graph, tf_session):
        super(NeuralSession, self).__init__(agent)
        self.kb = kb
        self.lexicon = lexicion  # For entity linking
        self.vocab = vocab  # For mapping from predictions to tokens

        # Generation params
        self.stop_symbols = map(self.vocab.to_ind, (END_TURN, END_UTTERANCE))
        self.max_len = 20

        # Tensorflow computation graph and session
        self.tf_graph = tf_graph
        self.tf_session = tf_session

        # Dialogue history
        self.entities = []  # Remember the entities we've seen at each position so that we can update the graph
        self.init_state = None  # Starting state for generation/encoding

    def receive(self, event):
        # TODO:
        # entity linking
        # encode, update init_state for send
        raise NotImplementedError

    def send(self):
        # TODO:
        # copy stuff in encdec.generate
        # update init_state
        raise NotImplementedError
