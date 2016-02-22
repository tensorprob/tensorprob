import tensorflow as tf

from .. import config
from ..distribution import Distribution


@Distribution
def Uniform(name=None):
    X = tf.placeholder(config.dtype, name=name)

    Distribution.logp = tf.constant(0, dtype=config.dtype)

    Distribution.integral = lambda lower, upper: tf.cast(upper, config.dtype) - tf.cast(lower, config.dtype)

    return X


@Distribution
def UniformInt(name=None):
    X = tf.placeholder(config.int_dtype, name=name)

    Distribution.logp = tf.constant(0, dtype=config.dtype)

    Distribution.integral = lambda lower, upper: tf.cast(tf.floor(upper) - tf.ceil(lower), config.dtype)

    return X
