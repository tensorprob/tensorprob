import tensorflow as tf

from .. import config
from ..distribution import Distribution


@Distribution
def Exponential(lambda_, name=None):
    X = tf.placeholder(config.dtype, name=name)

    Distribution.logp = tf.log(lambda_) - lambda_*X

    def integral(lower, upper):
        return tf.exp(-lambda_*lower) - tf.exp(-lambda_*upper)

    Distribution.integral = integral

    return X
