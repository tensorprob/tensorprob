import tensorflow as tf

from .. import config
from ..distribution import Distribution


@Distribution
def Uniform(name=None):
    X = tf.placeholder(config.dtype, name=name)

    Distribution.logp = tf.constant(0, dtype=config.dtype)

    Distribution.integral = lambda lower, upper: tf.cast(upper, tf.float64) - tf.cast(lower, tf.float64)

    return X


@Distribution
def UniformInt(name=None):
    X = tf.placeholder(config.int_dtype, name=name)

    Distribution.logp = tf.constant(1, dtype=config.int_dtype)

    Distribution.integral = lambda lower, upper: 1 / (upper - lower)

    return X
