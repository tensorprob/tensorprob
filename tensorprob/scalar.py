import numpy as np
from tensorflow import Variable

from . import config
from . import utilities


class Scalar(Variable):
    def __init__(self, name=None, dtype=config.dtype, lower=-np.inf, upper=np.inf):
        from .model import Model
        name = name or utilities.generate_name()
        self.lower = lower
        self.upper = upper

        Model.current_model.track_variable(self)

        super(Scalar, self).__init__(dtype(0.0), name=name)
