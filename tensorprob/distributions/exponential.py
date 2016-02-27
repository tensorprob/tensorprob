import tensorflow as tf

from .. import config
from ..distribution import Distribution


@Distribution
def Exponential(lambda_, name=None):
    X = tf.placeholder(config.dtype, name=name)

    Distribution.logp = tf.log(lambda_) - lambda_*X

    def integral(lower, upper):
        upper_integrand = tf.cond(
            tf.is_inf(tf.cast(upper, config.dtype)),
            lambda: tf.constant(1, dtype=config.dtype),
            lambda: tf.exp(-lambda_*upper)
        )

        lower_integrand = tf.cond(
            tf.is_inf(tf.cast(lower, config.dtype)),
            lambda: tf.constant(0, dtype=config.dtype),
            lambda: tf.exp(-lambda_*lower)
        )

        return lower_integrand - upper_integrand

    Distribution.integral = integral

    return X
