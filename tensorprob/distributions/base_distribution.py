import tensorflow as tf

from .. import config
from .. import utilities


class BaseDistribution(tf.Tensor):
    def __init__(self, name=None):
        from ..model import Model

        Model.current_model.track_variable(self)

        # Get the op of a new placeholder and use it as our op
        placeholder = tf.placeholder(dtype=config.dtype, name=name or utilities.generate_name())
        my_op = placeholder.op
        my_op.outputs.clear()
        my_op.outputs.append(self)

        super(BaseDistribution, self).__init__(my_op, 0, config.dtype)

    def logp(self):
        pass

    def pdf(self):
        pass

    def cdf(self, lim):
        pass

