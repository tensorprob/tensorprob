import numpy as np
import tensorflow as tf

from .. import config
from ..distribution import Distribution


def _normal_logp(X, mu, sigma):
    return (
        tf.log(1 / (tf.constant(np.sqrt(2 * np.pi), dtype=config.dtype) * sigma)) -
        (X - mu)**2 / (tf.constant(2, dtype=config.dtype) * sigma**2)
    )


def _normal_cdf(lim, mu, sigma):
    return 0.5 * tf.erfc((mu - lim) / (tf.constant(np.sqrt(2), config.dtype) * sigma))


@Distribution
def Normal(mu, sigma, name=None):
    # TODO(chrisburr) Just use NormalN?
    X = tf.placeholder(config.dtype, name=name)

    Distribution.logp = _normal_logp(X, mu, sigma)

    def integral(lower, upper):
        upper_integrand = tf.cond(
            tf.is_inf(tf.cast(upper, config.dtype)),
            lambda: tf.constant(1, dtype=config.dtype),
            lambda: _normal_cdf(upper, mu, sigma)
        )

        lower_integrand = tf.cond(
            tf.is_inf(tf.cast(lower, config.dtype)),
            lambda: tf.constant(0, dtype=config.dtype),
            lambda: _normal_cdf(lower, mu, sigma)
        )

        return upper_integrand - lower_integrand

    Distribution.integral = integral

    return X


# @Distribution
# def NormalN(mus, sigmas, name=None):
#     X = tf.placeholder(config.dtype, name=name)

#     logps = [_normal_logp(X, mu, sigma) for mu, sigma in zip(mus, sigmas)]

#     def cdf(lim):
#         raise NotImplementedError

#     Distribution.logp = sum(logps)
#     Distribution.integral = lambda lower, upper: cdf(upper) - cdf(lower)

#     return X
