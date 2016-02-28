import tensorflow as tf

class BaseSampler(object):

    def __init__(self, session=None):
        self._session = session or tf.get_default_session()

    def sample(self, variables, cost, gradient=None):
        raise NotImplementedError

    @property
    def session(self):
        return self._session

    @session.setter
    def session(self, session):
        self._session = session
