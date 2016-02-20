from .distributions import Uniform

# Parameter is just an alias for Uniform
class Parameter(Uniform):
    def __init__(self, lower=None, upper=None, name=None):
        super(Parameter, self).__init__(lower=lower, upper=upper, name=name)

