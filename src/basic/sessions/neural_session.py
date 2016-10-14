__author__ = 'anushabala'
from session import Session


class NeuralSession(Session):
    """
    NeuralSession represents a dialogue agent backed by a neural model. This class stores the knowledge graph,
    tensorflow session, and other model-specific information needed to update the model (when a message is received)
    or generate a response using the model (when send() is called).
    This class is closely related to but different from NeuralSystem. NeuralSystem is the class that actually loads the
    model from disk, while NeuralSession represents a specific instantiation of that model, and contains information
    (like the agent index and the knowledge base) that are specific to an agent in the current dialogue.
    """
    def __init__(self, agent, kb, lexicon, vocab, tf_graph, tf_session):
        super(NeuralSession, self).__init__(agent)
        self.kb = kb
        self.lexicon = lexicon  # For entity linking
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
