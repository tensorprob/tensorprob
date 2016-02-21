import tensorflow as tf

from .. import config
from ..distribution import Distribution


@Distribution
def Poisson(lambda_, name=None):
    k = tf.placeholder(config.int_dtype, name=name)

    Distribution.logp = k*lambda_ - lambda_ + tf.lgamma(k+1)

    # TODO Distribution.integral = ...

    return k
