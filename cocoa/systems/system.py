class System(object):
    """An abstract class for building a Session object.
    """
    def new_session(self, agent, kb):
        raise NotImplementedError

    @classmethod
    def name(cls):
        return 'base'
