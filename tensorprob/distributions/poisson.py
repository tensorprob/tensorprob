import tensorflow as tf

from .. import config
from ..distribution import Distribution


@Distribution
def Poisson(lambda_, name=None):
    k = tf.placeholder(config.int_dtype, name=name)

    # FIXME tf.lgamma only supports floats so cast before
    Distribution.logp = (
        tf.cast(k, config.dtype)*tf.log(lambda_) -
        lambda_ -
        tf.lgamma(tf.cast(k+1, config.dtype))
    )

    # TODO Distribution.integral = ...
    def integral(l, u):
        return tf.constant(1, dtype=config.dtype)
    Distribution.integral = integral

    return k
