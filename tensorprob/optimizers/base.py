
class BaseOptimizer(object):

    def __init__(self):
        raise NotImplementedError

    def minimize(self, cost, gradient=None, variables=None):
        raise NotImplementedError

    @property
    def session(self):
        return self._session

    @session.setter
    def session(self, session):
        self._session = session
