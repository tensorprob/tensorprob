from .. import utilities
from .. import config
import tensorflow as tf

class BaseDistribution(tf.Tensor):
    def __init__(self, name=None):
        # Get the op of a new placeholder and use it as our op
        placeholder = tf.placeholder(dtype=config.dtype, name=name or utilities.generate_name())
        my_op = placeholder.op
        my_op.outputs.clear()
        my_op.outputs.append(self)

        super(BaseDistribution, self).__init__(my_op, 0, config.dtype)
