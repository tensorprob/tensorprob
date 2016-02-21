import tensorflow as tf

from .. import config
from ..distribution import Distribution


@Distribution
def Uniform(name=None):
    X = tf.placeholder(config.dtype, name=name)

    Distribution.logp = tf.constant(1, dtype=config.dtype)

    Distribution.integral = lambda lower, upper: 1 / (upper - lower)

    return X
