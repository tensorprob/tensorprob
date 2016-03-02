import tensorflow as tf

from .. import config
from ..distribution import Distribution


@Distribution
def Uniform(name=None):
    X = tf.placeholder(config.dtype, name=name)

    Distribution.logp = tf.fill(tf.shape(X), config.dtype(0))

    def integral(lower, upper):
        return tf.cond(
            tf.logical_or(
                tf.is_inf(tf.cast(lower, config.dtype)),
                tf.is_inf(tf.cast(upper, config.dtype))
            ),
            lambda: tf.constant(1, dtype=config.dtype),
            lambda: tf.cast(upper, config.dtype) - tf.cast(lower, config.dtype),
        )

    Distribution.integral = integral

    return X


@Distribution
def UniformInt(name=None):
    X = tf.placeholder(config.int_dtype, name=name)

    Distribution.logp = tf.fill(tf.shape(X), config.dtype(0))

    def integral(lower, upper):
        val = tf.cond(
            tf.logical_or(
                tf.is_inf(tf.ceil(tf.cast(lower, config.dtype))),
                tf.is_inf(tf.floor(tf.cast(upper, config.dtype)))
            ),
            lambda: tf.constant(1, dtype=config.dtype),
            lambda: tf.cast(upper, config.dtype) - tf.cast(lower, config.dtype),
        )
        return val

    Distribution.integral = integral

    return X
