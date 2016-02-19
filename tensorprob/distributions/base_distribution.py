from .. import utilities


class BaseDistribution(object):
    def __init__(self, name=None):
        self.name = name or utilities.generate_name()
