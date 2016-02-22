
class BaseOptimizer(object):

    def __init__(self):
        raise NotImplementedError

    def minimize(self, cost, gradient=None, variables=None):
        raise NotImplementedError
