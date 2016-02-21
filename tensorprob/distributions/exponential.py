import tensorflow as tf

from .. import config
from ..distribution import Distribution


@Distribution
def Exponential(lambda_, name=None):
    X = tf.placeholder(config.dtype, name=name)

    Distribution.logp = tf.log(lambda_) - lambda_*X

    def cdf(lim):
        return tf.constant(1, dtype=config.dtype) - tf.exp(-lambda_*lim)

    Distribution.integral = lambda lower, upper: cdf(upper) - cdf(lower)

    return X
